from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tensorflow as tf
import numpy as np


from noise_variance_reader import NoiseVarianceReader
from noise_variance_model import NoiseVarianceModel, ModelStage
from utils import Utils
from tensorflow_utils import TensorflowUtils
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
        

    def _get_base_config(self):
        """Get model config."""
        config = {
            'table_name': 'month_noise_variance_test',
            'test_size' : 20,
            'init_scale': 0.1,
            'max_grad_norm': 5,
            'layer_size': 30,
            'max_epoch': 200,
            'keep_prob': 0.8,
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
            
            inputs, targets = reader.iterator.get_next()
            train_iterator_init_op = reader.get_iterator_initializer(config['batch_size'])
            test_iterator_init_op = reader.get_iterator_initializer(eval_config['batch_size'], test=True)
            
            with tf.variable_scope("Model", reuse=None, initializer=initializer): 
                m = NoiseVarianceModel(config=config, inputs=inputs, targets=targets, num_features=reader.num_features)
                            
            saver = tf.train.Saver()    
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=False)) as session:
                
                if tf.train.latest_checkpoint(save_path):
                    saver.restore(session, tf.train.latest_checkpoint(save_path))
                else:
                    session.run(tf.global_variables_initializer())
                
                train_writer = tf.summary.FileWriter(save_path, session.graph)
                
                train_error = 0
                learning_rate = 0
                global_step = 0
                
                for i in range(config['max_epoch']):
                    
                    train_error, predictions = self._run_epoch(session, m, iterator_init_op=train_iterator_init_op, eval_op=m.train_op, verbose=False)
                    learning_rate =  session.run(m.lr)
                    #print('Train Epoch: {:d} Mean Squared Error: {:.5f} Learning rate: {:.5f}'.format(i + 1, train_mse, learning_rate))
                    global_step = session.run(tf.train.get_global_step())
                    train_writer.add_summary(TensorflowUtils.summary_value('Training Loss', train_error), global_step)
                    #print('train predictions: ', predictions)
                    print('Train Step: {:d} Maximum likelihood cost: {:.5f} Learning rate: {:.5f}'.format(global_step, train_error, learning_rate))
                    
                    if (i + 1) % 50 == 0:
                        test_error, predictions = self._run_epoch(session, m, iterator_init_op=test_iterator_init_op)
                        #print('test predictions: ', predictions)
                        train_writer.add_summary(TensorflowUtils.summary_value('Test Loss', test_error), global_step)
                        print('Test Step: {:d} Maximum likelihood cost: {:.5f}'.format(global_step, test_error))
                    
                train_writer.close()
                
                
            
                name_dict = {'global_step':global_step, 'error':test_error}
                self._checkpoint(saver, session, save_path, True, **name_dict)
                    
    
    def _run_epoch(self, session, model, iterator_init_op, eval_op=None, verbose=False):
        
        
        fetches ={
            'cost': model.cost,
            'predictions': model.predictions
        }
        
        feed_dict={}
        
        if eval_op is not None:
            fetches['eval_op'] = eval_op
            feed_dict={model.keep_prob:self._config['keep_prob']}
        try:
            session.run(iterator_init_op)
            step = 1
            costs = 0.
            predictions = []
            while(True):
                vals = session.run(fetches, feed_dict=feed_dict)
                cost = vals['cost']
                predictions.append(vals['predictions'])
                costs += cost
                if verbose:
                    print('{:.3f} Maximum likelihood Cost: {:.5f}'.format(
                        step, 
                         costs/step))
                step += 1
                
        except tf.errors.OutOfRangeError:
            pass
        
        return costs/step, predictions
    
    def _checkpoint(self, saver, session, path, remove_current, **kwargs):
        file_name = 'model.ckpt'
        for key, value in kwargs.items():
            file_name += '.' + key +'_'+ str(value)
        
        if remove_current:
            Utils.remove_files_from_dir(path, ["model.ckpt", "checkpoint"])
        
        save_file = os.path.join(path, file_name)    
        saver.save(session, save_file)
            
                
modelTrainer = NoiseVarianceModelTrainer({'max_epoch' : 2000, 'batch_size':20})
modelTrainer.train()
            
        
        
        
        
        
        
        