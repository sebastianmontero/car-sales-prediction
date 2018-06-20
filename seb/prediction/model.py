from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as tfcudnn_rnn
import tensorflow.contrib.rnn as tfrnn
import tensorflow.contrib.estimator as tfestimator
import tensorflow.contrib.layers as tflayers
import numpy as np


from reader import Reader
from evaluator import Evaluator
import export_utils

from tensorflow.python.client import device_lib
from tensorflow.python.debug.wrappers.hooks import TensorBoardDebugHook

flags = tf.flags

flags.DEFINE_string('model', 'small', "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string('save_path', '/home/nishilab/Documents/python/model-storage/car-sales-prediction/save', "Model output directory")
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

FLAGS = flags.FLAGS
BASIC = 'basic'
CUDNN = 'cudnn'
BLOCK = 'block'

TRAIN = 'training'
VALIDATE = 'validate'
TEST = 'test'

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'gradient-descent': tf.train.GradientDescentOptimizer 
    }

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

        
class Model(object):
    
    def __init__(self, stage, config, generator):
        self._is_training = stage == TRAIN
        self._stage = stage
        self._rnn_params = None
        self._cell = None
        self._generator = generator
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps  
        
        inputs, targets = generator.get_data()
        targets = tf.reshape(targets, [-1, 1])
        output, state = self._build_rnn_graph(inputs, config, self._is_training)
        
        linear_w = tf.get_variable(
            'linear_w', 
            [config.layers[-1], 1],
            dtype=data_type(),
            initializer=tflayers.xavier_initializer())
        linear_b = tf.get_variable(
            'linear_b', 
            initializer=tf.random_uniform([1]), 
            dtype=data_type())
        
        output = tf.nn.xw_plus_b(output, linear_w, linear_b)
        self._predictions = output
        self._cost = tf.losses.mean_squared_error(targets, output, config.error_weight)
        
        self._final_state = state
            
        if not self._is_training:
            return
        self._lr = tf.Variable(config.learning_rate, trainable=False)
        optimizer = OPTIMIZERS[FLAGS.optimizer](self._lr)
        optimizer = tfestimator.clip_gradients_by_norm(optimizer, config.max_grad_norm)
        self._train_op = optimizer.minimize(self._cost, global_step=tf.train.get_or_create_global_step())
        self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)
    
    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)
    
    #TODO: Still need to change it so that it uses the config layers parameter instead of num_layers
    #and hidden size     
    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tfcudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units=config.hidden_size,
            input_size=config.hidden_size,
            dropout=1 - config.keep_prob if is_training else 0)
        self._rnn_params = tf.get_variable(
            'lstm_params', 
            initializer=tflayers.xavier_initializer(), 
            validate_shape=False)
        c = tf.zeros([config.num_layers, self.batch_size, self.hidden_size], tf.float32)
        h = tf.zeros([config.num_layers, self.batch_size, self.hidden_size], tf.float32)
        self._initial_state = (tfrnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.layers[-1]])
        return outputs,(tfrnn.LSTMStateTuple(h=h, c=c),)
    
    def _get_lstm_cell(self, rnn_mode, hidden_size, is_training):
        if rnn_mode == BASIC:
            return tfrnn.LSTMCell(
                hidden_size, 
                state_is_tuple=True,
                reuse = not is_training )
        if rnn_mode == BLOCK:
            return tfrnn.LSTMBlockCell(
                hidden_size)
        raise ValueError('rnn mode {} not supported'.format(rnn_mode))
    
    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        def make_cell(rnn_mode, hidden_size):
            cell = self._get_lstm_cell(rnn_mode, hidden_size, is_training)
            if is_training and config.keep_prob < 1:
                cell = tfrnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell
        cell = tfrnn.MultiRNNCell(
            [make_cell(config.rnn_mode, hidden_size) for hidden_size in config.layers], state_is_tuple=True)
        
        self._initial_state = cell.zero_state(config.batch_size, data_type())
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state, time_major=True)
        output = tf.reshape(outputs, [-1, config.layers[-1]])
        return output, state      
        
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    
    def export_ops(self, name):
        self._name = name
        ops = {export_utils.with_prefix(self._name, 'cost'): self._cost,
               export_utils.with_prefix(self._name, 'predictions'): self._predictions}
        
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = export_utils.with_prefix(self._name, 'initial')
        self._final_state_name = export_utils.with_prefix(self._name, 'final')
        export_utils.export_state_tuples(self._initial_state, self._initial_state_name)
        export_utils.export_state_tuples(self._final_state, self._final_state_name)
    
    def import_ops(self):
        if self._is_training:
            self._train_op = tf.get_collection_ref('train_op')[0]
            self._lr = tf.get_collection_ref('lr')[0]
            self._new_lr = tf.get_collection_ref('new_lr')[0]
            self._lr_update = tf.get_collection_ref('lr_update')[0]
            rnn_params = tf.get_collection_ref('rnn_params')
            if self._cell and rnn_params:
                params_saveable = tfcudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope='Model/RNN')
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(export_utils.with_prefix(self._name, 'cost'))[0]
        self._predictions = tf.get_collection_ref(export_utils.with_prefix(self._name, 'predictions'))[0]
        num_replicas = FLAGS.num_gpus if self._name == 'Train' else 1
        self._initial_state = export_utils.import_state_tuples(self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = export_utils.import_state_tuples(self._final_state, self._final_state_name, num_replicas)
        
    
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
    def generator(self):
        return self._generator
    
    @property
    def predictions(self):
        return self._predictions
    

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    max_grad_norm = 5
    num_layers = 2
    num_steps = 12
    hidden_size = 100
    max_epoch = 100
    keep_prob = 1.0
    lr_decay = 0.98
    mse_not_improved_threshold = 3
    batch_size = 1
    rnn_mode = BLOCK
    layers = [100]
    error_weight = 1000000 
    

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
    rnn_mode = BLOCK
    layers = [650, 650]
    error_weight = 100000

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
    rnn_mode = BLOCK
    layers = [1500, 1500]
    error_weight = 100000


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
    rnn_mode = BLOCK
    layers = [2]
    error_weight = 100000
        
        
def run_epoch(session, model, config, eval_op=None, verbose=False, vocabulary=None):
    
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

def get_config():
    """Get model config."""
    config = None
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "medium":
        config = MediumConfig()
    elif FLAGS.model == "large":
        config = LargeConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
        config.rnn_mode = BASIC
    
    config.learning_rate = float(FLAGS.learning_rate)
    
    return config    
        

def main(_):
    
    gpus = [
        x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'
    ]
    
    if FLAGS.num_gpus > len(gpus):
        raise ValueError('Your machine only has {} gpus'.format(len(gpus)))
    
    line_id = 13
    window_size = 52
    reader = Reader(line_id, window_size, ['interest_rate', 'exchange_rate', 'consumer_confidence_index'])
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    test_predictions = []
    #eval_config.num_steps = 1
    
    while reader.next_window():
        print()
        print('Window from: {} to {}'.format(reader.get_start_month_id(), reader.get_end_month_id()))
        print()
        save_path = os.path.join(FLAGS.save_path, reader.get_window_name())  
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            
            with tf.name_scope('Train'):
                
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    generator = reader.get_generator(config.batch_size, config.num_steps) 
                    m = Model(stage = TRAIN, config=config, generator=generator)
                tf.summary.scalar('Training Loss', m.cost)
                tf.summary.scalar('Learning Rate', m.lr)
            
            with tf.name_scope('Test'):
                generator = reader.get_generator(eval_config.batch_size, eval_config.num_steps, for_test=True)
                with tf.variable_scope('Model', reuse=True, initializer=initializer):
                    mtest = Model(stage=TEST, config=eval_config, generator=generator)
                    
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
                model.import_ops()
            config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
            hooks = [
                #TensorBoardDebugHook('localhost:6064')
            ]
            with tf.train.MonitoredTrainingSession(
                checkpoint_dir=save_path, 
                config=config_proto,
                hooks=hooks) as session:
                
                min_mse = None
                mse_not_improved_count = 0
                
                for i in range(config.max_epoch):
                    
                    train_mse, predictions = run_epoch(session, m, config, eval_op=m.train_op, verbose=False)
                    learning_rate =  session.run(m.lr)
                    print('Train Epoch: {:d} Mean Squared Error: {:.5f} Learning rate: {:.5f}'.format(i + 1, train_mse, learning_rate))
                    
                    if min_mse is None or train_mse < min_mse:
                        min_mse = train_mse
                        mse_not_improved_count = 0
                    else:
                        mse_not_improved_count += 1
                        
                    if mse_not_improved_count > config.mse_not_improved_threshold:
                        learning_rate = learning_rate * config.lr_decay
                        m.assign_lr(session, learning_rate)
                            
                test_mse, predictions = run_epoch(session, mtest, config)
                test_predictions.append(predictions[-1])
                print('Test Mean Squared Error: {:.5f}'.format(test_mse))
                evaluator = Evaluator(reader, predictions, reader.get_end_window_pos(True))
                evaluator.plot_real_target_vs_predicted()
                #evaluator.plot_scaled_target_vs_predicted()
                evaluator.plot_real_errors()
                #evaluator.plot_scaled_errors()
    evaluator = Evaluator(reader, test_predictions, -1)
    #evaluator.plot_real_target_vs_predicted()
    #evaluator.plot_scaled_target_vs_predicted()
            
                
if __name__ == '__main__':
    tf.app.run()
            
        
        
        
        
        
        
        