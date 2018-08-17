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
    
    '''def _get_raw_data(self):
        sql = ("select * " + 
               "from " + self._table_name)
        data = pd.read_sql(sql, con=DBManager.get_engine())
        data.reindex(np.random.permutation(data.index))
        return data'''
    
    
    def _get_raw_data(self):
        x = np.linspace(0, 1, 1000)
        y = x ** 2
        return pd.DataFrame({'x':x,'variance':y})
    
    def get_generator(self, batch_size, test=False):
        if test:
            start = self._train_num_rows
            end = self._num_rows
        else:
            start = 0
            end = self._train_num_rows
            
        data = self._raw_data.iloc[start:end]
        x = data[self._input_cols].values
        y = data[self.target_col].values
        return NoiseVarianceGenerator(x, y, batch_size)
    
'''reader = NoiseVarianceReader('month_noise_variance_test', 20)

generator = reader.get_generator(5)
x,y = generator.get_data()

with tf.Session() as sess:
    for i in range(10):
        vals = sess.run({'x':x, 'y': y})
        print('x value:')
        print(vals['x'])
        print('')
        print('')
        print('y value:')
        print(vals['y'])


print('---------TEST--------------------')

generator = reader.get_generator(1, test=True)
x,y = generator.get_data()

with tf.Session() as sess:
    for i in range(10):
        vals = sess.run({'x':x, 'y': y})
        print('x value:')
        print(vals['x'])
        print('')
        print('')
        print('y value:')
        print(vals['y'])'''
