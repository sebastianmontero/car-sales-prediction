from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np


from reader import Reader
from evaluator import Evaluator
from model import Model, ModelRNNMode, ModelStage
import export_utils

from tensorflow.python.client import device_lib
from tensorflow.python.debug.wrappers.hooks import TensorBoardDebugHook

flags = tf.flags

flags.DEFINE_string('model', 'small', "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string('save_path', '/home/nishilab/Documents/python/model-storage/car-sales-prediction/save/', "Model output directory")
flags.DEFINE_string('output_file', '/home/nishilab/Documents/python/model-storage/language-modeling/test-output.txt', "File where the words produced by test will be saved")
flags.DEFINE_bool('use_fp16', False, "Train using 16 bits floats instead of 32 bits")
flags.DEFINE_integer('num_gpus', 1, 
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string('rnn_mode', None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, lstm, "
                    "and lstm_block_cell classes.")
flags.DEFINE_string('optimizer', 'adagrad',
                    "The optimizer to use: adam, adagrad, gradient-descent. "
                    "Default is adagrad.")
flags.DEFINE_string('learning_rate', "1.0", #adam requires smaller learning rate
                    "The starting learning rate to use"
                    "Default is 0.1")

flags.DEFINE_string('node-ip-address', None, #adam requires smaller learning rate
                    "Node ip address")
flags.DEFINE_string('object-store-name', None, #adam requires smaller learning rate
                    "Object store name")
flags.DEFINE_string('object-store-manager-name', None, #adam requires smaller learning rate
                    "object-store-manager-name")
flags.DEFINE_string('local-scheduler-name', None, #adam requires smaller learning rate
                    "local-scheduler-name")
flags.DEFINE_string('redis-address', None, #adam requires smaller learning rate
                    "redis-address")

FLAGS = flags.FLAGS

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'gradient-descent': tf.train.GradientDescentOptimizer 
    }

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    max_grad_norm = 5
    num_layers = 2
    num_steps = 12
    hidden_size = 100
    max_epoch = 1000
    keep_prob = 1
    lr_decay = 0.98
    mse_not_improved_threshold = 3
    batch_size = 1
    rnn_mode = ModelRNNMode.BLOCK
    layers = [100]
    error_weight = 1000000
    data_type = tf.float32
    

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    mse_not_improved_threshold = 3
    batch_size = 20
    rnn_mode = ModelRNNMode.BLOCK
    layers = [650, 650]
    error_weight = 100000
    data_type = tf.float32

class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    mse_not_improved_threshold = 3
    batch_size = 20
    rnn_mode = ModelRNNMode.BLOCK
    layers = [1500, 1500]
    error_weight = 100000
    data_type = tf.float32


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    mse_not_improved_threshold = 3
    batch_size = 20
    rnn_mode = ModelRNNMode.BLOCK
    layers = [2]
    error_weight = 100000
    data_type = tf.float32
        

class ModelTrainer():
    
    def __init__(self, config):
        self._config = None
        self._eval_config = None
        self._reader = None
        self._save_path = FLAGS.save_path
        self._setup(config)
        
    @property
    def save_path(self):
        return self._save_path
    
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
            predictions.append(np.reshape(vals['predictions'], [-1, model.batch_size]))
            costs += cost
            if verbose:
                print('{:.3f} Mean Squared Error: {:.5f}'.format(
                    step * 1.0 / epoch_size, 
                     np.exp(costs/step)))
        
        predictions = np.split(np.concatenate(predictions), model.batch_size,axis=1)
        predictions = np.reshape(np.concatenate(predictions), [-1,1])       
        return costs, predictions

    def _get_base_config(self):
        """Get model config."""
        config = {
            'init_scale': 0.1,
            'max_grad_norm': 5,
            'num_layers': 2,
            'num_steps': 12,
            'hidden_size': 100,
            'max_epoch': 100,
            'keep_prob': 1,
            'lr_decay': 0.98,
            'mse_not_improved_threshold': 3,
            'batch_size': 1,
            'rnn_mode': ModelRNNMode.BLOCK,
            'layers': [100],
            'error_weight': 1000000,
            'data_type': tf.float32,
            'included_features': ['interest_rate', 'exchange_rate', 'consumer_confidence_index']
        }
        
        if FLAGS.rnn_mode:
            config['rnn_mode'] = FLAGS.rnn_mode
        if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
            config['rnn_mode'] = ModelRNNMode.BASIC
        
        config['learning_rate'] = float(FLAGS.learning_rate)
        config['optimizer'] = OPTIMIZERS[FLAGS.optimizer]
        
        return config    
            
    def _setup(self, config):
        
        gpus = [
            x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'
        ]
        
        if FLAGS.num_gpus > len(gpus):
            raise ValueError('Your machine only has {} gpus'.format(len(gpus)))
        
        
        line_id = 13
        window_size = 52
        self._config = self._get_base_config()
        self._config.update(config)
        self._eval_config = self._config.copy()
        self._eval_config['batch_size'] = 1
        self._reader = Reader(line_id, window_size, self._config['included_features'])
        
    
    def train(self):
        
        assert (self._config), "setup has to be called before calling train"
        
        test_predictions = []
        #eval_config.num_steps = 1
        reader = self._reader
        config = self._config
        eval_config = self._eval_config
        
        reader.reset()
        
        while reader.next_window():
            print()
            print('Window from: {} to {}'.format(reader.get_start_month_id(), reader.get_end_month_id()))
            print()
            save_path = os.path.join(FLAGS.save_path, reader.get_window_name())
            save_file = os.path.join(save_path, 'model.ckpt')  
            with tf.Graph().as_default():
                initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])
                
                with tf.name_scope('Train'):
                    
                    with tf.variable_scope("Model", reuse=None, initializer=initializer):
                        generator = reader.get_generator(config['batch_size'], config['num_steps']) 
                        m = Model(stage=ModelStage.TRAIN, config=config, generator=generator)
                    tf.summary.scalar('Training Loss', m.cost)
                    tf.summary.scalar('Learning Rate', m.lr)
                
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
                tf.train.import_meta_graph(metagraph)
                for model in models.values():
                    model.import_ops(FLAGS.num_gpus)
                    
                saver = tf.train.Saver()
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=soft_placement)) as session:
                    
                    if tf.train.latest_checkpoint(save_path):
                        saver.restore(session, tf.train.latest_checkpoint(save_path))
                    else:
                        session.run(tf.global_variables_initializer())
                    
                    merged = tf.summary.merge_all()
                    train_writer = tf.summary.FileWriter(save_path, session.graph)
                    
                    min_mse = None
                    mse_not_improved_count = 0
                    
                    for i in range(config['max_epoch']):
                        
                        train_mse, predictions = self._run_epoch(session, m, eval_op=m.train_op, verbose=False)
                        learning_rate =  session.run(m.lr)
                        print('Train Epoch: {:d} Mean Squared Error: {:.5f} Learning rate: {:.5f}'.format(i + 1, train_mse, learning_rate))
                        train_writer.add_summary(session.run(merged), session.run(tf.train.get_global_step()))
                        
                        if min_mse is None or train_mse < min_mse:
                            min_mse = train_mse
                            mse_not_improved_count = 0
                        else:
                            mse_not_improved_count += 1
                            
                        if mse_not_improved_count > config['mse_not_improved_threshold']:
                            learning_rate = learning_rate * config['lr_decay']
                            m.assign_lr(session, learning_rate)
                                
                    test_mse, predictions = self._run_epoch(session, mtest)
                    test_predictions.append(predictions[-1])
                    print('Test Mean Squared Error: {:.5f}'.format(test_mse))
                    evaluator = Evaluator(reader, predictions, reader.get_end_window_pos(True))
                    print("Absolute Mean Error: {:.2f} Relative Mean Error: {:.2f}%".format(evaluator.real_absolute_mean_error(), evaluator.real_relative_mean_error()))
                    #print("Absolute Mean Error: {:.2f}".format(evaluator.real_absolute_mean_error()))
                    #evaluator.plot_real_target_vs_predicted()
                    #evaluator.plot_scaled_target_vs_predicted()
                    #evaluator.plot_real_errors()
                    #evaluator.plot_scaled_errors()
                    saver.save(session, save_file, tf.train.get_global_step())
        evaluator = Evaluator(reader, test_predictions, -1)
        #evaluator.plot_real_target_vs_predicted()
        #evaluator.plot_real_errors()
        #print("Absolute Mean Error: {:.2f}".format(evaluator.real_absolute_mean_error()))
        print("Absolute Mean Error: {:.2f} Relative Mean Error: {:.2f}%".format(evaluator.real_absolute_mean_error(), evaluator.real_relative_mean_error()))
        #evaluator.plot_scaled_target_vs_predicted()
        return evaluator
            
                
'''modelTrainer = ModelTrainer({})
modelTrainer.train()'''
            
        
        
        
        
        
        
        