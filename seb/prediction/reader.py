'''
Created on Jun 11, 2018

@author: nishilab
'''
import os
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from utils import Utils
from generator import Generator
from db_manager import DBManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Reader(object):
    
    PREDICTED_VARS_START_POS = 2
    
    scaler_fit_domain_values = {
        'sales': [0, 180],
        'interest_rate': [1.0, 10.0],
        'exchange_rate': [8.0, 18.0],
        'consumer_confidence_index': [70,120],
        'manufacturing_confidence_index': [70,120],
        'economic_activity_index': [80, 115]
    }
    

    def __init__(self, line_id, window_size, included_features, prediction_size = 1):
        assert (window_size > 0), "Window size must be greater than zero"
        self._line_id = str(line_id) 
        self._window_size = window_size
        self._window_pos = -1
        self._prediction_size = prediction_size
        self._included_features = included_features
        self._scale_features = ['sales'] + included_features
        self._features = ['month_of_year_sin', 'month_of_year_cos', 'sales']
        self._features.extend(self._included_features)
        self._num_features = len(self._features)
        self._predicted_vars = self._scale_features
        self._num_predicted_vars = len(self._predicted_vars)
        self._init_fleeting_vars()
        
    def _init_fleeting_vars(self):
        self._engine = DBManager.get_engine()
        self._scaler = MinMaxScaler((-1,1))
        self._data = None
        self._start_month_id = None
        self._raw_data = None
        self._iterator = None
        self._inputs = None
        self._targets = None
        self._process_data()
        self._num_windows = self._data.shape[0] - self._window_size
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_engine']
        del state['_scaler']
        del state['_data']
        del state['_raw_data']
        del state['_start_month_id']
        del state['_num_windows']
        del state['_iterator']
        del state['_inputs']
        del state['_targets']
        return state 
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_fleeting_vars()

    @property
    def num_features(self):
        return self._num_features
    
    @property
    def window_pos(self):
        return self._window_pos
    
    @property
    def predicted_vars(self):
        return self._predicted_vars
    
    @property
    def num_predicted_vars(self):
        return self._num_predicted_vars
    
    @property
    def iterator(self):
        if self._iterator is None:
            self._prepare_iterator()
        return self._iterator
    
    def _prepare_iterator(self):
        self._iterator = tf.data.Iterator.from_structure(self._get_dataset_types(),((None, None, self._num_features),(None, None, self._num_predicted_vars))) 
    
    def get_iterator_elements(self):
        if self._inputs is None:
            self._inputs, self._targets = self.iterator.get_next()
        return self._inputs, self._targets
        
    def get_predicted_var_name(self, pos):
        return self._predicted_vars[pos]
    
    def _get_raw_data(self):
        
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
        
        self._raw_data = data_df = self._get_raw_data()
        assert (data_df.shape[0] >= (self._window_size + 1)), 'Data length: {} is smaller than window size + 1 (Test Value): {}'.format(data_df.shape[0], (self._window_size + 1))
         
        self._start_month_id = int(data_df['month_id'][0])
        
        data_np = data_df[self._scale_features].values.astype('float') #get non month cols 
        
        month_np = self._process_month(data_df[['month_id']].copy())
        
        fit_data_np = data_np[:self._window_size]
        fit_data_np = self._add_scaler_fit_domain_values(self._scale_features, fit_data_np)
        self._scaler.fit(fit_data_np)
        data_np = self._scaler.transform(data_np)
        
        data_np = np.concatenate((month_np, data_np), axis=1)
        
        self._data = pd.DataFrame(data_np, columns=self._features, dtype=np.float32)
         
        
    def _add_scaler_fit_domain_values(self, features, data):
        
        max_length = self._get_scaler_fit_domain_values_max_length()
        domain_values = []
        for i, feature in enumerate(features):
            if feature in self.scaler_fit_domain_values:
                vals = self.scaler_fit_domain_values[feature][:]
                if len(vals) < max_length:
                    vals.extend(self._get_sample(max_length - len(vals), data, i))
                domain_values.append(vals)
            else:
                domain_values.append(self._get_sample(max_length, data, i))
                
        domain_values = np.array(domain_values).transpose()
        return np.concatenate([data,domain_values])
    
    def _get_sample(self, num_values, data, col):
        sample = []
        for i in np.random.randint(len(data), size=num_values):
            sample.append(data[i][col])
        return sample
    
    def _get_scaler_fit_domain_values_max_length(self):
        max_length = 0
        for key,value in self.scaler_fit_domain_values.items():
            if len(value) > max_length:
                max_length = len(value)
        return max_length
    
    def _process_month(self, data_df):
        data_df['month_of_year'] = data_df['month_id'].apply(lambda x: Utils.month_id_to_month_of_year(x))
        data_df['month_of_year_sin'] = data_df['month_of_year'].apply(lambda x: math.sin(x))
        data_df['month_of_year_cos'] = data_df['month_of_year'].apply(lambda x: math.cos(x))
        return data_df.values[:,-2:]
    
    def next_window(self):
        self._window_pos += 1
        return self.has_more_windows()
        
    def _get_data(self, for_test = False):
        return self._get_window_data(self._data, self._window_pos, for_test) 
    
    def get_data(self, end_window_pos, length, scaled = False):
        return self._get_window_data_by_end_pos(self.get_all_data(scaled), end_window_pos, length)
    
    def get_all_data(self, scaled=False):
        return self._data if scaled else self._raw_data
    
    def _get_window_data_by_end_pos(self, source, end_window_pos, length):
        end_window_pos = self.process_absolute_pos(end_window_pos)
        if length < 0:
            length = end_window_pos
        assert (end_window_pos <= source.shape[0]), "end_window_pos index out of bounds"
        assert (length <= end_window_pos), "length must be lower than end_window_pos"
        return source[end_window_pos - length: end_window_pos]
    
    def process_absolute_pos(self, pos):
        return pos if pos >= 0 else self._data.shape[0] + pos + 1
    
    def _get_window_data(self, source, window_pos, for_test = False):
        assert (window_pos >= 0), "Next window must be called first to get data for window"
        return source[self._window_pos: self.get_end_window_pos(for_test) ].copy()
    
    def has_more_windows(self):
        return self._window_pos < self._num_windows
    
    def get_iterator_initializer(self, batch_size, num_steps, for_test=False):
        
        data = self._get_data(for_test).values
        length =  data.shape[0]
        residual = (length - self._prediction_size) % batch_size
        data = data[residual:]
        num_batches = (length - self._prediction_size) // batch_size
        epoch_size = math.ceil((length - self._prediction_size) / (num_steps * batch_size))
        ds = self._get_dataset(data, batch_size, num_batches)
        ds = ds.batch(num_steps).repeat()
        ii = self.iterator.make_initializer(ds)
        return ii, epoch_size
    
    def _get_dataset_types(self):
        ds = self._get_dataset(self._data[0:2].values, 1, 1)
        return ds.output_types
    
    def _get_dataset(self, data, batch_size, num_batches):
        x_data = []
        y_data = []
        for num_batch in range(num_batches):
            x_batch = []
            y_batch = []
            for batch_pos in range(batch_size):
                pos =  batch_pos * num_batches + num_batch
                x_batch.append(data[pos])
                y_batch.append(data[pos + self._prediction_size][self.PREDICTED_VARS_START_POS:])
            x_data.append(x_batch)
            y_data.append(y_batch)
        return tf.data.Dataset.from_tensor_slices((np.asarray(x_data), np.asarray(y_data)))
    
    def _get_absolute_pos(self, delta=0):
        return self._window_pos + delta
    
    def get_window_name(self, for_test=False):
        return 'w-{}-{}'.format(self.get_start_month_id(), self.get_end_month_id(for_test))
    
    def get_end_window_pos(self, for_test=False):
        return self._window_pos + self._window_size + (1 if for_test else 0)
    
    def get_start_month_id(self):
        return Utils.add_months_to_month_id(self._start_month_id, self._window_pos)
    
    def get_end_month_id(self, for_test=False):
        return Utils.add_months_to_month_id(self._start_month_id, self.get_end_window_pos(for_test))
    
    def unscale_features(self, features, round_sales=True):
        features = np.array(features)
        scaled_features = features[:,:len(self._scale_features)]
        unscaled = self._scaler.inverse_transform(scaled_features)
        if round_sales:
            for l in unscaled:
                l[0] = round(max(l[0],0))
        return np.concatenate((unscaled, features[:,len(self._scale_features):]), axis=1).tolist()
        
        
    
    def reset(self):
        self._window_pos = -1
        


'''features = ['interest_rate', 'exchange_rate', 'energy_price_index_roc_prev_month','energy_price_index_roc_start_year']
#features = ['inflation_index_roc_prev_month']
reader = Reader(13, 36, features)

print(reader.get_data(36, 36))

reader.next_window()

generator = reader.get_generator(1, 40, False)
x, y = generator.get_data()

with tf.Session() as sess:
    #for i in range(4):
    vals = sess.run({'x':x, 'y': y})
    print('x value:')
    print(vals['x'])
    print('')
    print('')
    #x_vals = np.reshape(vals['x'], (-1, 7))
    #print(np.array(reader.unscale_features(np.take(x_vals, [2,3,4,5,6], axis=1), round_sales=True)))
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
        
        
'''features = ['inflation_index_roc_prev_month',
                                   'consumer_confidence_index']
#features = ['inflation_index_roc_prev_month']
reader = Reader(13, 37, features)

while reader.next_window():
    
    print(reader.get_start_month_id(), reader.get_end_month_id(True))

    x, y = reader.get_iterator_elements()
    
    with tf.Session() as sess:
        ii,_ = reader.get_iterator_initializer(1, 40, False)
        sess.run(ii)
        #for i in range(4):
        vals = sess.run({'x':x, 'y': y})
        print('x value:')
        print(vals['x'][-2:])
        print('')
        #x_vals = np.reshape(vals['x'], (-1, 7))
        #print(np.array(reader.unscale_features(np.take(x_vals, [2,3,4,5,6], axis=1), round_sales=True)))
        #print('')
        print('y value:')
        print(vals['y'][-2:])
        print()
        print()
    
    with tf.Session() as sess:
        ii,_ = reader.get_iterator_initializer(1, 40, True)
        sess.run(ii)
        #for i in range(4):
        vals = sess.run({'x':x, 'y': y})
        print('x value:')
        print(vals['x'][-2:])
        print('')
        #x_vals = np.reshape(vals['x'], (-1, 7))
        #print(np.array(reader.unscale_features(np.take(x_vals, [2,3,4,5,6], axis=1), round_sales=True)))
        #print('')
        print('y value:')
        print(vals['y'][-2:])
        print()
        print()'''
