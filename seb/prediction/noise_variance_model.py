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
    
    def __init__(self, stage, config, generator):
        self._is_training = stage == ModelStage.TRAIN
        self._stage = stage
        self._generator = generator
        
        inputs, targets = generator.get_data()
        output = self._build_model(inputs, config)
        self._predictions = output
        self._cost = tf.losses.mean_squared_error(targets, output)
        #self._cost = tf.reduce_sum(tf.log(output) + (targets/output)) / 2
                
        if not self._is_training:
            return
        
        self._optimize(config, self._cost)
        
    def _build_model(self, inputs, config):
        
        w1 = tf.Variable(tf.truncated_normal([self._generator.num_features, config['layer_size']], dtype=tf.float64), name='WeightsLayer1')
        b1 = tf.Variable(tf.zeros([config['layer_size']],dtype=tf.float64), name='BiasesLayer1')
        h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(inputs, w1) + b1), config['keep_prob'])
        w2 = tf.Variable(tf.truncated_normal([config['layer_size'], 1], dtype=tf.float64), name='WeightsLayer2')
        b2 = tf.Variable(tf.zeros([1], dtype=tf.float64), name='BiasesLayer1')
        output = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)
        output = tf.reshape(output, [-1])
        return output
    
    def _optimize(self, config, cost):
        self._lr = tf.Variable(config['learning_rate'], trainable=False)
        optimizer = config['optimizer'](self._lr)
        optimizer = tfestimator.clip_gradients_by_norm(optimizer, config['max_grad_norm'])
        self._train_op = optimizer.minimize(cost, global_step=tf.train.get_or_create_global_step())
        
        
    def export_ops(self, name):
        self._name = name
        ops = {export_utils.with_prefix(self._name, 'cost'): self._cost,
               export_utils.with_prefix(self._name, 'predictions'): self._predictions}
        
        if self._is_training:
            ops.update(lr=self._lr)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
    
    def import_ops(self):
        if self._is_training:
            self._train_op = tf.get_collection_ref('train_op')[0]
            self._lr = tf.get_collection_ref('lr')[0]
        self._cost = tf.get_collection_ref(export_utils.with_prefix(self._name, 'cost'))[0]
        self._predictions = tf.get_collection_ref(export_utils.with_prefix(self._name, 'predictions'))[0]
        

    @property
    def cost(self):
        return self._cost
        
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
    def generator(self):
        return self._generator
    
    @property
    def predictions(self):
        return self._predictions
        