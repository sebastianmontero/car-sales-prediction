from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.estimator as tfestimator
from enum import Enum

import export_utils

class ModelStage(Enum):
    TRAIN = 'training'
    VALIDATE = 'validate'
    TEST = 'test'

        
class NoiseVarianceModel(object):
    
    def __init__(self, config, inputs, targets, num_features):
        self._num_features = num_features
        
        output = self._build_model(inputs, config)
        self._predictions = output
        self._cost = tf.losses.mean_squared_error(targets, output)
        #self._cost = tf.reduce_sum(tf.log(output) + (targets/output)) / 2
                
        self._optimize(config, self._cost)
        
    def _build_model(self, inputs, config):
        
        self._keep_prob = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float64), shape=())
        
        w1 = tf.Variable(tf.truncated_normal([self._num_features, config['layer_size']], dtype=tf.float64), name='WeightsLayer1')
        b1 = tf.Variable(tf.zeros([config['layer_size']],dtype=tf.float64), name='BiasesLayer1')
        h = tf.nn.dropout(tf.nn.relu(tf.matmul(inputs, w1) + b1), self._keep_prob)
        w2 = tf.Variable(tf.truncated_normal([config['layer_size'], config['layer_size']], dtype=tf.float64), name='WeightsLayer1')
        b2 = tf.Variable(tf.zeros([config['layer_size']],dtype=tf.float64), name='BiasesLayer1')
        h = tf.nn.dropout(tf.nn.relu(tf.matmul(h, w2) + b2), self._keep_prob)
        w3 = tf.Variable(tf.truncated_normal([config['layer_size'], 1], dtype=tf.float64), name='WeightsLayer2')
        b3 = tf.Variable(tf.zeros([1], dtype=tf.float64), name='BiasesLayer1')
        output = tf.nn.sigmoid(tf.matmul(h, w3) + b3)
        output = tf.reshape(output, [-1])
        return output
    
    def _optimize(self, config, cost):
        self._lr = tf.Variable(config['learning_rate'], trainable=False)
        optimizer = config['optimizer'](self._lr)
        optimizer = tfestimator.clip_gradients_by_norm(optimizer, config['max_grad_norm'])
        self._train_op = optimizer.minimize(cost, global_step=tf.train.get_or_create_global_step())
            
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def cost(self):
        return self._cost
    
    @property
    def lr(self):
        return self._lr
    
    @property
    def train_op(self):
        return self._train_op
    
    @property
    def predictions(self):
        return self._predictions
    