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


class Reader(object):

    def __init__(self, line_id, window_size):
        self._engine = self._connect_to_db()
        self._line_id = str(line_id) 
        self._window_size = window_size
        self.window_pos = -1
        self._window_step_size = 1
        self._included_features = ['interest_rate']
        self._features = ['month_of_year_sin', 'month_of_year_cos', 'sales']
        self._features.extend(self._included_features)
        self._num_features = len(self._features)
        self._scaler = MinMaxScaler((-1,1))
        self._data = None
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
        data_np = data_df.values[: , 1:] #get non month cols
        month_np = self._process_month(data_df)
        self._scaler.fit(data_np[:self._window_size, :])
        data_np = self._scaler.transform(data_np)
        data_np = np.concatenate((month_np, data_np), axis=1)
        self._data = pd.DataFrame(data_np, columns=self._features)
    
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
        self.window_pos += 1
        return self.has_more_windows()
        
    def _get_data(self):
        return self._data.iloc[self.window_pos: self.window_pos + self._window_size]
        
    def has_more_windows(self):
        return self.window_pos < self._num_windows
    
    def get_generator(self, batch_size, num_steps):
        return Generator(self._get_data().values, batch_size, num_steps)
        
        
'''reader = Reader(13, 37)

while reader.next_window():
    print(reader._get_data())

reader.next_window()

generator = reader.get_generator(3, 12)

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

    
