from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tensorflow as tf
import numpy as np


from noise_variance_reader import NoiseVarianceReader
from noise_variance_model import NoiseVarianceModel, ModelStage
from utils import Utils
import export_utils

from tensorflow.python.client import device_lib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'gradient-descent': tf.train.GradientDescentOptimizer 
    }        

class NoiseVarianceModelTrainer():
    
    def __init__(self, config):
        self._config = None
        self._eval_config = None
        self._reader = None
        self._save_path = None
        self._setup(config)
        
    
    def _run_epoch(self, session, model, eval_op=None, verbose=False):
        
        costs = 0.
        predictions = []
        
        fetches ={
            'cost': model.cost,
            'predictions': model.predictions
        }
        
        if eval_op is not None:
            fetches['eval_op'] = eval_op
        
        epoch_size = model.generator.epoch_size
        for step in range(1, epoch_size + 1):
            vals = session.run(fetches)
            cost = vals['cost']
            predictions.append(vals['predictions'])
            costs += cost
            if verbose:
                print('{:.3f} Maximum likelihood Cost: {:.5f}'.format(
                    step * 1.0 / epoch_size, 
                     costs/step))
        
        return costs/epoch_size, predictions

    def _get_base_config(self):
        """Get model config."""
        config = {
            'table_name': 'month_noise_variance_test',
            'test_size' : 20,
            'init_scale': 0.1,
            'max_grad_norm': 5,
            'layer_size': 15,
            'max_epoch': 200,
            'keep_prob': 1.0,
            'learning_rate': 0.1,
            'lr_decay': 0.98,
            'mse_not_improved_threshold': 3,
            'batch_size': 5,
            'optimizer': OPTIMIZERS['adagrad'],
            'save_path': '/home/nishilab/Documents/python/model-storage/noise-variance-model/'
        }    
        return config    
            
    def _setup(self, config):
        
        self._config = self._get_base_config()
        self._config.update(config)
        self._eval_config = self._config.copy()
        self._eval_config['batch_size'] = 1
        self._reader = NoiseVarianceReader(self._config['table_name'], self._config['test_size'])        
    
    def train(self):
        
        #eval_config.num_steps = 1
        reader = self._reader
        config = self._config
        eval_config = self._eval_config
        

        tf.reset_default_graph()
        save_path = config['save_path']
        best_save_path = os.path.join(save_path, 'best')  
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])
            
            with tf.name_scope('Train'):
                
                with tf.variable_scope("Model", reuse=None, initializer=initializer): 
                    m = NoiseVarianceModel(stage=ModelStage.TRAIN, config=config, generator=reader.get_generator(config['batch_size']))
                tf.summary.scalar('Training Loss', m.cost)
                tf.summary.scalar('Learning Rate', m.lr)
            
            with tf.name_scope('Test'):
                with tf.variable_scope('Model', reuse=True, initializer=initializer):
                    mtest = NoiseVarianceModel(stage=ModelStage.TEST, config=eval_config, generator=reader.get_generator(eval_config['batch_size'], test=True))
                    
            '''models = {'Train': m, 'Valid': mvalid, 'Test': mtest}'''
            models = {'Train': m, 'Test': mtest}
            for name, model in models.items():
                model.export_ops(name)
            metagraph = tf.train.export_meta_graph()
        
        
        with tf.Graph().as_default():
            
            saver = tf.train.import_meta_graph(metagraph)
            for model in models.values():
                model.import_ops()
                
            #saver = tf.train.Saver()    
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=False)) as session:
                
                if tf.train.latest_checkpoint(save_path):
                    saver.restore(session, tf.train.latest_checkpoint(save_path))
                else:
                    session.run(tf.global_variables_initializer())
                
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(save_path, session.graph)
                
                train_error = 0
                learning_rate = 0
                global_step = 0
                
                for i in range(config['max_epoch']):
                    
                    train_error, predictions = self._run_epoch(session, m, eval_op=m.train_op, verbose=False)
                    learning_rate =  session.run(m.lr)
                    #print('Train Epoch: {:d} Mean Squared Error: {:.5f} Learning rate: {:.5f}'.format(i + 1, train_mse, learning_rate))
                    global_step = session.run(tf.train.get_global_step())
                    train_writer.add_summary(session.run(merged), global_step)
                    #print('train predictions: ', predictions)
                    print('Train Step: {:d} Maximum likelihood cost: {:.5f} Learning rate: {:.5f}'.format(global_step, train_error, learning_rate))
                
                train_writer.close()
                test_error, predictions = self._run_epoch(session, mtest)
                
                #print('test predictions: ', predictions)
                print('Test Step: {:d} Maximum likelihood cost: {:.5f}'.format(global_step, test_error))
                
            
                name_dict = {'global_step':global_step, 'error':test_error}
                self._checkpoint(saver, session, save_path, True, **name_dict)
                    
    
    def _checkpoint(self, saver, session, path, remove_current, **kwargs):
        file_name = 'model.ckpt'
        for key, value in kwargs.items():
            file_name += '.' + key +'_'+ str(value)
        
        if remove_current:
            Utils.remove_files_from_dir(path, ["model.ckpt", "checkpoint"])
        
        save_file = os.path.join(path, file_name)    
        saver.save(session, save_file)
            
                
modelTrainer = NoiseVarianceModelTrainer({'max_epoch' : 200, 'batch_size':20})
modelTrainer.train()
            
        
        
        
        
        
        
        