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
from ensemble_reporter import EnsembleReporter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class IncompatibleBaseEnsemble(Exception):
    pass

class Reader(object):
    
    scaler_fit_domain_values = {
        'sales': [0, 180],
        'interest_rate': [1.0, 10.0],
        'exchange_rate': [8.0, 18.0],
        'consumer_confidence_index': [70,120],
        'manufacturing_confidence_index': [70,120],
        'economic_activity_index': [80, 115]
    }
    

    def __init__(self, line_id, window_size, included_features, prediction_size = 1, base_ensembles=[]):
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
        self._base_ensembles = self._load_base_ensembles(base_ensembles)
        self._init_fleeting_vars()
        
    def _init_fleeting_vars(self):
        self._engine = DBManager.get_engine()
        self._scaler = MinMaxScaler((-1,1))
        self._data = None
        self._start_month_id = None
        self._raw_data = None
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
        return state 
    
    def __setstate__(self, state):
        
        if '_dont_scale_features' in state:
            state['_scale_features'] = np.concatenate((['sales'],  state['_included_features']))
            state['_predicted_vars'] = ['sales']
            state['_num_predicted_vars'] = 1
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
    
    def _load_base_ensembles(self, paths):
        ensembles = []
        for path in paths:
            print('Loading base ensemble: {} {} ...'.format(path, self._line_id))
            ensembleReporter = EnsembleReporter(path, overwrite=True)
            ensemble = ensembleReporter.get_ensemble_evaluator(find_best_ensemble=True)
            if not self._is_base_ensemble_compatible(ensemble):
                raise IncompatibleBaseEnsemble('Base ensemble is not compatible with current reader')
            ensembles.append(ensemble)
            print('Loaded base ensemble: {} ...'.format(path))
        return ensembles
    
    def _is_base_ensemble_compatible(self, ensemble):
        ereader = ensemble.reader
        return ereader.predicted_vars == self.predicted_vars
        
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
    
    def get_generator(self, batch_size, num_steps, for_test=False):
        data = self._set_base_predictions(self._get_data(for_test), for_test).values
        return Generator(data, batch_size, num_steps, self._num_predicted_vars, self._prediction_size)
    
    def _set_base_predictions(self, data, for_test=False):
        num_predictions = len(self._base_ensembles)
        pos = data.shape[0] - num_predictions - (1 if for_test else 0)
        for ensemble in self._base_ensembles:
            predictions = ensemble.predictions_by_absolute_pos(self._get_absolute_pos(pos), scaled=True)
            if predictions is not None:
                data.iloc[pos][self._predicted_vars] = predictions
            pos += 1
        return data
    
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
        num_sf = len(self._scale_features)
        scaled_features = features[:,:num_sf]
        
        if self._num_predicted_vars < num_sf:
            scaled_features = np.concatenate((scaled_features, np.zeros((len(features), num_sf - self._num_predicted_vars))), axis=1)
        
        unscaled = self._scaler.inverse_transform(scaled_features)
        unscaled = unscaled[:, :self._num_predicted_vars]
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
reader = Reader(13, 37, features, base_ensembles=['/home/nishilab/Documents/python/model-storage/ensemble-run-nationwide_sf_ifp_1m-20180829152705465891', '/home/nishilab/Documents/python/model-storage/ensemble-run-nationwide_sf_ifp_2m-20180903224408157528'])
#reader = Reader(13, 37, features)

while reader.next_window():
    
    print(reader.get_start_month_id(), reader.get_end_month_id(True))

    generator = reader.get_generator(1, 40, True)
    x, y = generator.get_data()
    
    with tf.Session() as sess:
        #for i in range(4):
        vals = sess.run({'x':x, 'y': y})
        print('x value:')
        print(vals['x'][-3:])
        print('')
        #x_vals = np.reshape(vals['x'], (-1, 7))
        #print(np.array(reader.unscale_features(np.take(x_vals, [2,3,4,5,6], axis=1), round_sales=True)))
        #print('')
        print('y value:')
        print(vals['y'][-3:])
        print()
        print()'''
    
