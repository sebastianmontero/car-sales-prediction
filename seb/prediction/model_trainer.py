from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import argparse
import tensorflow as tf
import numpy as np


from reader import Reader
from model import Model, ModelRNNMode, ModelStage
from utils import Utils
import export_utils

from tensorflow.python.client import device_lib

from tensorflow.python.debug.wrappers.hooks import TensorBoardDebugHook
from evaluator import Evaluator
from storage_manager import StorageManager, StorageManagerType, PickleAction
from tensorflow_utils import TensorflowUtils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='small', type=str, required=False, help="A type of model. Possible options are: small, medium, large.")
parser.add_argument('--use_fp16', default=False, type=bool, required=False, help="Train using 16 bits floats instead of 32 bits")
parser.add_argument('--num_gpus', default=0, type=int, required=False, help="If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
parser.add_argument('--rnn_mode', default=None, type=str, required=False, help="The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, lstm, "
                    "and lstm_block_cell classes.")
parser.add_argument('--optimizer', default='adagrad', type=str, required=False, help="The optimizer to use: adam, adagrad, gradient-descent.")
parser.add_argument('--learning_rate', default=1.0, type=float, required=False, help="The starting learning rate to use"
                    "Default is 0.1")

FLAGS,_ = parser.parse_known_args()

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'gradient-descent': tf.train.GradientDescentOptimizer 
    }        

class ModelTrainer():
    
    def __init__(self, config):
        self._config = None
        self._eval_config = None
        self._reader = None
        self._save_path = None
        self._setup(config)
        
    
    def _run_epoch(self, session, model, eval_op=None, verbose=False):
        
        costs = 0.
        predictions = []
        state = session.run(model.initial_state)
        
        fetches ={
            'cost': model.cost,
            'final_state': model.final_state,
            'predictions': model.predictions
        }
        
        if eval_op is not None:
            fetches['eval_op'] = eval_op
        
        epoch_size = model.generator.epoch_size
        for step in range(1, epoch_size + 1):
            feed_dict = {}
            for i, (c,h) in enumerate(model.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
            vals = session.run(fetches, feed_dict)
            cost = vals['cost']
            state = vals['final_state']
            predictions.append(np.reshape(vals['predictions'], [-1, model.batch_size, model.generator.num_predicted_vars]))
            
            costs += cost
            if verbose:
                print('{:.3f} Mean Squared Error: {:.5f}'.format(
                    step * 1.0 / epoch_size, 
                     costs/step))
        
        predictions = np.split(np.concatenate(predictions), model.batch_size,axis=1)
        predictions = np.reshape(np.concatenate(predictions), [-1,model.generator.num_predicted_vars])       
        return costs, predictions

    def _get_base_config(self):
        """Get model config."""
        config = {
            'prediction_size' : 1,
            'train_months' : 51, #Window size is generated by train_months + prediction_size
            'line_id': 13,
            'init_scale': 0.1,
            'max_grad_norm': 5,
            'num_layers': 2,
            'num_steps': 72,
            'hidden_size': 10,
            'max_epoch': 10,
            'keep_prob': 1,
            'lr_decay': 0.98,
            'mse_not_improved_threshold': 3,
            'batch_size': 1,
            'rnn_mode': ModelRNNMode.BLOCK,
            'layers': [70],
            'error_weight': 1000000,
            'data_type': tf.float32,
            'save_path': '/home/nishilab/Documents/python/model-storage/car-sales-prediction/save/',
            'included_features': ['inflation_index_roc_prev_month',
                                  'consumer_confidence_index',
                                  'energy_price_index_roc_prev_month'],
            'store_window' : True # Whether the configuration and evaluator object should be saved for every window
        }
        
        if FLAGS.rnn_mode:
            config['rnn_mode'] = FLAGS.rnn_mode
        if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
            config['rnn_mode'] = ModelRNNMode.BASIC
        
        config['learning_rate'] = FLAGS.learning_rate
        config['optimizer'] = OPTIMIZERS[FLAGS.optimizer]
        
        return config    
            
    def _setup(self, config):
        
        gpus = [
            x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'
        ]
        
        if FLAGS.num_gpus > len(gpus):
            raise ValueError('Your machine only has {} gpus'.format(len(gpus)))
        
        
        #line_id = 13
        #line_id = 102
        #line_id=201
        self._config = self._get_base_config()
        self._config.update(config)
        self._config_layers(self._config)
        self._eval_config = self._config.copy()
        self._eval_config['batch_size'] = 1
        if 'window_size' not in self._config:
            self._config['window_size'] = self._config['train_months'] + self._config['prediction_size'] 
        self._reader = Reader(self._config['line_id'], self._config['window_size'], self._config['included_features'], self._config['prediction_size'])
    
    def _config_layers(self, config):
                
        i = 0
        key = 'layer_0'
        layers = []
        
        while key in config and config[key]:
            layers.append(config[key])
            i += 1
            key = 'layer_' + str(i)
        
        if len(layers):
            config['layers'] = layers
            
    
    def train(self):
        
        test_predictions = []
        #eval_config.num_steps = 1
        reader = self._reader
        config = self._config
        eval_config = self._eval_config
        evaluator_sm = StorageManager.get_storage_manager(StorageManagerType.EVALUATOR)
        config_sm = StorageManager.get_storage_manager(StorageManagerType.CONFIG)
        reader.reset()
        
        while reader.next_window():
            print()
            print('Window from: {} to {}'.format(reader.get_start_month_id(), reader.get_end_month_id()))
            print()
            tf.reset_default_graph()
            save_path = os.path.join(config['save_path'], reader.get_window_name())
            best_save_path = os.path.join(save_path, 'best')  
            with tf.Graph().as_default():
                initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])
                
                with tf.name_scope('Train'):
                    
                    with tf.variable_scope("Model", reuse=None, initializer=initializer):
                        generator = reader.get_generator(config['batch_size'], config['num_steps']) 
                        m = Model(stage=ModelStage.TRAIN, config=config, generator=generator)
                
                with tf.name_scope('Test'):
                    generator = reader.get_generator(eval_config['batch_size'], eval_config['num_steps'], for_test=True)
                    with tf.variable_scope('Model', reuse=True, initializer=initializer):
                        mtest = Model(stage=ModelStage.TEST, config=eval_config, generator=generator)
                        
                '''models = {'Train': m, 'Valid': mvalid, 'Test': mtest}'''
                models = {'Train': m, 'Test': mtest}
                for name, model in models.items():
                    model.export_ops(name)
                metagraph = tf.train.export_meta_graph()
                if tf.__version__ < '1.1.0' and FLAGS.num_gpus > 1:
                    raise ValueError('Your version of tensorflow does not support more than 1 gpu')
                
                soft_placement = False
                
                if FLAGS.num_gpus > 1:
                    soft_placement = True
                    export_utils.auto_parallel(metagraph, m)
            
            
            with tf.Graph().as_default():
                
                test_absolute_error_tf = tf.Variable(-1.0, trainable=False, name='test_absolute_error')
                mse_not_improved_count_tf = tf.Variable(0, trainable=False, name='mse_not_improved_count')
                min_mse_tf = tf.Variable(-1, trainable=False, name='min_mse')
                saver = tf.train.import_meta_graph(metagraph)
                for model in models.values():
                    model.import_ops(FLAGS.num_gpus)
                    
                #saver = tf.train.Saver()    
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=soft_placement)) as session:
                    
                    if tf.train.latest_checkpoint(save_path):
                        saver.restore(session, tf.train.latest_checkpoint(save_path))
                    else:
                        session.run(tf.global_variables_initializer())
                    
                    train_writer = tf.summary.FileWriter(save_path, session.graph)
                    
                    min_mse = session.run(min_mse_tf)
                    mse_not_improved_count = session.run(mse_not_improved_count_tf)
                    train_mse = 0
                    learning_rate = 0
                    global_step = 0
                    
                    for i in range(config['max_epoch']):
                        
                        train_mse, predictions = self._run_epoch(session, m, eval_op=m.train_op, verbose=False)
                        learning_rate =  session.run(m.lr)
                        #print('Train Epoch: {:d} Mean Squared Error: {:.5f} Learning rate: {:.5f}'.format(i + 1, train_mse, learning_rate))
                        global_step = session.run(tf.train.get_global_step())
                        train_writer.add_summary(TensorflowUtils.summary_value('Train Loss', train_mse), global_step)
                        train_writer.add_summary(TensorflowUtils.summary_value('Learning Rate', train_mse), global_step)
                        
                        if min_mse < 0 or train_mse < min_mse:
                            min_mse = train_mse
                            min_mse_tf.load(min_mse, session)
                            mse_not_improved_count = 0
                            
                        else:
                            mse_not_improved_count += 1
                        
                        mse_not_improved_count_tf.load(mse_not_improved_count, session)
                        
                            
                        if mse_not_improved_count > config['mse_not_improved_threshold']:
                            learning_rate = learning_rate * config['lr_decay']
                            m.assign_lr(session, learning_rate)
                            
                    
                    
                    print('Train Step: {:d} Mean Squared Error: {:.5f} Learning rate: {:.5f}'.format(global_step, train_mse, learning_rate))            
                    test_mse, predictions = self._run_epoch(session, mtest)
                    test_predictions.append(predictions[-1])
                    train_writer.add_summary(TensorflowUtils.summary_value('Test Loss', test_mse), global_step)
                    train_writer.close()
                    #print('Test Mean Squared Error: {:.5f}'.format(test_mse))
                    evaluator = Evaluator(reader, predictions, reader.get_end_window_pos(True), global_step)
                    #for pos,_ in enumerate(evaluator.predicted_vars):
                    #    evaluator.plot_target_vs_predicted(pos)
                    #print(predictions)
                    current_test_absolute_error = evaluator.window_real_absolute_mean_error()
                    best_test_absolute_error = session.run(test_absolute_error_tf)
                
                    name_dict = {'global_step':global_step, 'error':current_test_absolute_error}
                    if best_test_absolute_error == -1 or current_test_absolute_error < best_test_absolute_error:
                        print('Saving best model...')
                        
                        if config['store_window']:
                            evaluator_sm.pickle(evaluator, best_save_path, current_test_absolute_error)
                            config_sm.pickle(config, best_save_path, current_test_absolute_error)
                        
                        test_absolute_error_tf.load(current_test_absolute_error, session)
                        self._checkpoint(saver, session, best_save_path, True, **name_dict)
                    
                    self._checkpoint(saver, session, save_path, True, **name_dict)
                    
        evaluator = Evaluator(reader, test_predictions, -1, global_step)
        evaluator_sm.pickle(evaluator, config['save_path'], evaluator.absolute_mean_error(), PickleAction.BEST)
        config_sm.pickle(config, config['save_path'], evaluator.absolute_mean_error(), PickleAction.BEST)
        print()
        print("Absolute Mean Error: {:.2f} Relative Mean Error: {:.2f}%".format(evaluator.absolute_mean_error(), evaluator.relative_mean_error()))
        print()
        return evaluator
    
    def _checkpoint(self, saver, session, path, remove_current, **kwargs):
        file_name = 'model.ckpt'
        for key, value in kwargs.items():
            file_name += '.' + key +'_'+ str(value)
        
        if remove_current:
            Utils.remove_files_from_dir(path, ["model.ckpt", "checkpoint"])
        
        save_file = os.path.join(path, file_name)    
        saver.save(session, save_file)
            
                
modelTrainer = ModelTrainer({'max_epoch' : 1000, 'line_id':13, 'train_months':51, 'prediction_size':1, 'store_window':True})
modelTrainer.train()
            
        
        
        
        
        
        
        