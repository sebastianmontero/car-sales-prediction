from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as tfcudnn_rnn
import tensorflow.contrib.rnn as tfrnn
import tensorflow.contrib.estimator as tfestimator
import tensorflow.contrib.layers as tflayers
from enum import Enum

import export_utils

class ModelStage(Enum):
    TRAIN = 'training'
    VALIDATE = 'validate'
    TEST = 'test'

class ModelRNNMode(Enum):
    BASIC = 'basic'
    CUDNN = 'cudnn'
    BLOCK = 'block'

        
class Model(object):
    
    
    def __init__(self, stage, config, inputs, targets, num_predicted_vars):
        self._is_training = stage == ModelStage.TRAIN
        self._stage = stage
        self._rnn_params = None
        self._cell = None
        self.num_predicted_vars = num_predicted_vars
        self.batch_size = config['batch_size']
        self.num_steps = config['num_steps']  
        
        targets = tf.reshape(targets, [-1, num_predicted_vars])
        output, state = self._build_rnn_graph(inputs, config, self._is_training)
        
        linear_w = tf.get_variable(
            'linear_w', 
            [config['layers'][-1], num_predicted_vars],
            dtype=config['data_type'],
            initializer=tflayers.xavier_initializer())
        linear_b = tf.get_variable(
            'linear_b', 
            initializer=tf.random_uniform([num_predicted_vars]), 
            dtype=config['data_type'])
        
        output = tf.nn.xw_plus_b(output, linear_w, linear_b)
        self._predictions = output
        self._cost = tf.losses.mean_squared_error(targets, output, config['error_weight'])
        
        self._final_state = state
            
        if not self._is_training:
            return
        self._lr = tf.Variable(config['learning_rate'], trainable=False)
        optimizer = config['optimizer'](self._lr)
        optimizer = tfestimator.clip_gradients_by_norm(optimizer, config['max_grad_norm'])
        self._train_op = optimizer.minimize(self._cost, global_step=tf.train.get_or_create_global_step())
        self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)
    
    def _build_rnn_graph(self, inputs, config, is_training):
        if config['rnn_mode'] == ModelRNNMode.CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)
    
    #TODO: Still need to change it so that it uses the config layers parameter instead of num_layers
    #and hidden size     
    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tfcudnn_rnn.CudnnLSTM(
            num_layers=config['num_layers'],
            num_units=config['hidden_size'],
            input_size=config['hidden_size'],
            dropout=1 - config['keep_prob'] if is_training else 0)
        self._rnn_params = tf.get_variable(
            'lstm_params', 
            initializer=tflayers.xavier_initializer(), 
            validate_shape=False)
        c = tf.zeros([config['num_layers'], self.batch_size, self.hidden_size], tf.float32)
        h = tf.zeros([config['num_layers'], self.batch_size, self.hidden_size], tf.float32)
        self._initial_state = (tfrnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config['layers'][-1]])
        return outputs,(tfrnn.LSTMStateTuple(h=h, c=c),)
    
    def _get_lstm_cell(self, rnn_mode, hidden_size, is_training):
        if rnn_mode == ModelRNNMode.BASIC:
            return tfrnn.LSTMCell(
                hidden_size, 
                state_is_tuple=True,
                reuse = not is_training )
        if rnn_mode == ModelRNNMode.BLOCK:
            return tfrnn.LSTMBlockCell(
                hidden_size)
        raise ValueError('rnn mode {} not supported'.format(rnn_mode))
    
    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        def make_cell(rnn_mode, hidden_size):
            cell = self._get_lstm_cell(rnn_mode, hidden_size, is_training)
            if is_training and config['keep_prob'] < 1:
                cell = tfrnn.DropoutWrapper(
                    cell, output_keep_prob=config['keep_prob'])
            return cell
        cell = tfrnn.MultiRNNCell(
            [make_cell(config['rnn_mode'], hidden_size) for hidden_size in config['layers']], state_is_tuple=True)
        
        self._initial_state = cell.zero_state(config['batch_size'], config['data_type'])
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state, time_major=True)
        output = tf.reshape(outputs, [-1, config['layers'][-1]])
        return output, state      
        
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
         
    
    @property
    def initial_state(self):
        return self._initial_state
    
    @property
    def cost(self):
        return self._cost
    
    
    @property
    def final_state(self):
        return self._final_state
    
    @property
    def stage(self):
        return self._stage
    
    @property
    def lr(self):
        return self._lr
    
    @property
    def train_op(self):
        return self._train_op
    
    @property
    def initial_state_name(self):
        return self._initial_state_name
    
    @property
    def final_state_name(self):
        return self._final_state_name
    
    @property
    def predictions(self):
        return self._predictions
    
      
        
        
        
        
        
        
        