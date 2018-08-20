'''
Created on Jun 11, 2018

@author: nishilab
'''
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import Utils
from db_manager import DBManager
from noise_variance_generator import NoiseVarianceGenerator


class NoiseVarianceReader(object):
    
    target_col = 'variance'
    
    def __init__(self, table_name, test_size):
        self._table_name = table_name
        self._test_size = test_size
        self._raw_data = self._get_raw_data()
        self._num_rows = self._raw_data.shape[0]
        self._train_num_rows = self._num_rows - (math.floor(self._num_rows * (test_size / 100)))
        self._input_cols = [col for col in self._raw_data.columns if col != self.target_col]
        self._num_features = len(self._input_cols)
        self._iterator = None
    '''def _get_raw_data(self):
        sql = ("select * " + 
               "from " + self._table_name)
        data = pd.read_sql(sql, con=DBManager.get_engine())
        data.reindex(np.random.permutation(data.index))
        return data'''
    
    @property
    def iterator(self):
        if self._iterator is None:
            self._prepare_iterator()
        return self._iterator
    
    @property
    def num_features(self):
        return self._num_features
    
    def _get_raw_data(self):
        x = np.linspace(0, 1, 20)
        y = x ** 2
        return pd.DataFrame({'x':x,'variance':y})
    
    def _prepare_iterator(self):
        self._datasets = {
            'train': self._get_dataset(),
            'test': self._get_dataset(test=True)
        }
        self._iterator = tf.data.Iterator.from_structure(self._datasets['train'].output_types) 
    
    def _get_dataset(self, test=False):
        if test:
            start = self._train_num_rows
            end = self._num_rows
        else:
            start = 0
            end = self._train_num_rows
            
        data = self._raw_data.iloc[start:end]
        x = data[self._input_cols].values
        y = data[self.target_col].values
        return tf.data.Dataset.from_tensor_slices((x, y))
    
    def get_iterator_initializer(self, batch_size, test=False):
        iterator = self.iterator
        ds = 'test' if test else 'train'
        self._datasets[ds] = self._datasets[ds].batch(batch_size).shuffle(buffer_size = 1000)
        return iterator.make_initializer(self._datasets[ds], name=ds)
    
'''reader = NoiseVarianceReader('month_noise_variance_test', 20)

train_iterator_init = reader.get_iterator_initializer(5)
test_iterator_init = reader.get_iterator_initializer(1, test=True)

x,y = reader.iterator.get_next()

with tf.Session() as sess:
    for i in range(3):
        sess.run(train_iterator_init)
        try:    
            while(True):
                vals = sess.run({'x':x, 'y': y})
                print('x value:')
                print(vals['x'])
                print('y value:')
                print(vals['y'])
                print('')
                print('')
        except tf.errors.OutOfRangeError:
            print('epoch end')

print('---------TEST--------------------')

with tf.Session() as sess:
    for i in range(3):
        sess.run(test_iterator_init)
        try:    
            while(True):
                vals = sess.run({'x':x, 'y': y})
                print('x value:')
                print(vals['x'])
                print('y value:')
                print(vals['y'])
                print('')
                print('')
        except tf.errors.OutOfRangeError:
            print('epoch end')'''