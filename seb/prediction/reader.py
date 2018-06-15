'''
Created on Jun 11, 2018

@author: nishilab
'''

import math
import configparser
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils import Utils
from generator import Generator

from matplotlib.pyplot import axis

import tensorflow as tf


class Reader(object):

    def __init__(self, line_id, window_size):
        assert (window_size > 0), "Window size must be greater than zero"
        self._engine = self._connect_to_db()
        self._line_id = str(line_id) 
        self._window_size = window_size
        self._window_pos = -1
        self._window_step_size = 1
        self._included_features = ['interest_rate']
        self._features = ['month_of_year_sin', 'month_of_year_cos', 'sales']
        self._features.extend(self._included_features)
        self._num_features = len(self._features)
        self._scaler = MinMaxScaler((-1,1))
        self._sales_scaler = MinMaxScaler((-1,1))
        self._data = None
        self._start_month_id = None
        self._process_data()
        self._num_windows = self._data.shape[0] - window_size 
        

    @property
    def num_features(self):
        return self._num_features
    
    def _raw_data(self):
        
        included_features_str = (',' if len(self._included_features) else '') + ','.join(self._included_features) 
        
        sql = ("select mif.month_id, " + 
                       "mls.sales " +
                       included_features_str + " "
               "from month_input_features mif INNER JOIN "
                    "month_line_sales mls ON mif.month_id = mls.month_id "
               "where line_id = " + self._line_id + " "
                "order by month_id asc")
        return pd.read_sql(sql, con=self._engine)
    
    def _process_data(self):
        
        data_df = self._raw_data()
        assert (data_df.shape[0] > (self._window_size + 1)), 'Data length: {} is smaller than window size + 1: {}'.format(data_df.shape[0], (self._window_size + 1))
         
        self._start_month_id = int(data_df['month_id'][0])
        sales_np = data_df.values[:, 1:2]
        data_np = data_df.values[:, 2:] #get non month cols
        month_np = self._process_month(data_df)
        self._sales_scaler.fit(sales_np[:self._window_size])
        self._scaler.fit(data_np[:self._window_size])
        sales_np = self._sales_scaler.transform(sales_np)
        data_np = self._scaler.transform(data_np)
        data_np = np.concatenate((month_np, sales_np, data_np), axis=1)
        self._data = pd.DataFrame(data_np, columns=self._features, dtype=np.float32)
    
    def _process_month(self, data_df):
        data_df['month_of_year'] = data_df['month_id'].apply(lambda x: Utils.month_id_to_month_of_year(x))
        data_df['month_of_year_sin'] = data_df['month_of_year'].apply(lambda x: math.sin(x))
        data_df['month_of_year_cos'] = data_df['month_of_year'].apply(lambda x: math.cos(x))
        return data_df.values[:,-2:]
        
    def _connect_to_db(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        return create_engine(config['DB']['connection_url'])
    
    def next_window(self):
        self._window_pos += 1
        return self.has_more_windows()
        
    def _get_data(self, for_test = False):
        assert (self._window_pos >= 0), "Next window must be called first to get data for window"
        return self._data.iloc[self._window_pos: self.get_end_window_pos(for_test) ]
        
    def has_more_windows(self):
        return self._window_pos < self._num_windows
    
    def get_generator(self, batch_size, num_steps, for_test=False):
        return Generator(self._get_data(for_test).values, batch_size, num_steps)
    
    def get_window_name(self, for_test=False):
        return 'w-{}-{}'.format(self.get_start_month_id(), self.get_end_month_id(for_test))
    
    def get_end_window_pos(self, for_test=False):
        return self._window_pos + self._window_size + (self._window_step_size if for_test else 0)
    
    def get_start_month_id(self):
        return Utils.add_months_to_month_id(self._start_month_id, self._window_pos)
    def get_end_month_id(self, for_test=False):
        return Utils.add_months_to_month_id(self._start_month_id, self.get_end_window_pos(for_test))
    
    def unscale_sales(self, sales):
        return self._sales_scaler.inverse_transform(sales)
        
'''reader = Reader(13, 12)


reader.next_window()

generator = reader.get_generator(2, 3, False)
x, y = generator.get_data()

with tf.Session() as sess:
    for i in range(4):
        vals = sess.run({'x':x, 'y': y})
        print('x value:')
        print(vals['x'])
        print('')
        print('')
        print('y value:')
        print(vals['y'])

stage = 0
 

print(generator._data)
while generator.next_epoch_stage():
    print ('stage: {}'.format(stage))
    data_x, data_y = generator.get_stage()
    
    for i in data_x:
        print(i)
    
    print('')
    for i in data_y:
        print(i)
    
    stage += 1'''

    
