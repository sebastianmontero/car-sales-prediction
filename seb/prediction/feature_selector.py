from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import datetime
import ray
import pprint
from ray.tune import grid_search, run_experiments, register_trainable
from ray.tune.async_hyperband import AsyncHyperBandScheduler
from ray.tune.hyperband import HyperBandScheduler 

from storage_manager import StorageManager, StorageManagerType
from model_trainable import ModelTrainable
from ray.tune.variant_generator import grid_search


class FeatureSelector():
    
    '''FEATURES = ['consumer_confidence_index',
                'exchange_rate',
                'interest_rate',
                'manufacturing_confidence_index',
                'economic_activity_index',
                'energy_price_index_roc_prev_month',
                'energy_price_index_roc_start_year',
                'inflation_index_roc_prev_month',
                'inflation_index_roc_start_year']'''
    
    FEATURES = ['consumer_confidence_index',
                'energy_price_index_roc_prev_month']
    
    def __init__(self, config, max_features=9, repeats = 3):        
        assert (max_features <= len(self.FEATURES)), "max_features {} should be less than the number of possible features {}".format(max_features, len(self.FEATURES))
        self._repeats = repeats 
        self._max_features = max_features
        config['store_window'] = False
        self._config = config
        self._current_selected_features = []
        self._pprint = pprint.PrettyPrinter()
        self._config_sm = StorageManager.get_storage_manager(StorageManagerType.CONFIG)
        self._best_configs = None
        self._best_config = None
        self._base_path = os.path.join(ModelTrainable.BASE_PATH, self._get_experiments_base_dir())
        self._config['save_path'] = self._base_path
        
    
    @property
    def best_configs(self):
        return self._best_configs
    
    @property
    def best_config(self):
        return self._best_config
    
    def _feature_search_space(self, current_selected_features):
        free_features = [feature for feature in self.FEATURES if feature not in current_selected_features]
        space = []
        for feature in free_features:
            space_item = [feature]
            space_item.extend(current_selected_features)
            space.append(space_item)
        
        return space
    
    def _get_experiment_name(self, num_features):
        return 'feature-selection-{}-{}'.format(str(num_features), datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
    
    def _get_experiments_base_dir(self):
        return 'feature-selection-{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
    
    def _get_best_config(self, experiment_path):
        configs_dict = {}
        configs_errors = self._config_sm.get_objects_errors(experiment_path, recursive=True, sorted_=False)
        for config_error in configs_errors:
            key = str(config_error['obj']['included_features'])
            if key not in configs_dict:
                config_error['count'] = 1
                configs_dict[key] = config_error
            else:
                configs_dict[key]['error'] += config_error['error']
                configs_dict[key]['count'] += 1
        
        configs_list = []
        for key, obj in configs_dict.items():
            obj['avg_error'] = obj['error'] / obj['count']
            configs_list.append(obj)
        
        pprint.pprint(configs_list)
        return self._find_best_config(configs_list)
    
    def _find_best_config(self, configs):
        best =  None
        for config in configs:
            if best is None or config['avg_error'] < best['avg_error']:
                best = config
        
        return best
    
    def run(self):
        best_configs = []
        for num_features in range(1, self._max_features + 1):
            config = self._config.copy()
            config['included_features'] = grid_search(self._feature_search_space(self._current_selected_features))
            experiment_name = self._get_experiment_name(num_features)
            experiment_path = os.path.join(self._base_path, experiment_name) 
            
            run_experiments({
                experiment_name : {
                    'run': 'car_sales_prediction_trainable',
                    'trial_resources': {'cpu': 8, 'gpu': 1},
                    #'stop': {'neg_mean_loss': 0, 'training_iteration': 200},
                    'stop': {'training_iteration': 400},
                    'config' : config,
                    'repeat':self._repeats,
                }
            })
            best_config = self._get_best_config(experiment_path)
            self._current_selected_features = best_config['obj']['included_features']
            best_configs.append(best_config)
        
        self._best_configs = best_configs
        self._best_config = self._find_best_config(best_configs)
        
        print('Finished feature search!')
        
        self.print_best_config()
        self.print_best_configs()
    
    def print_best_config(self):
        print()
        print("Best overall configuration is:")
        print()
        self._pprint.pprint(self.best_config)
        print()
    
    def print_best_configs(self):
        print()
        print("Best configurations per number of features:")
        print()
        self._pprint.pprint(self.best_configs)
        print()
    

ray.init()
register_trainable('car_sales_prediction_trainable', ModelTrainable)

feature_selector = FeatureSelector({
                'keep_prob' : 1,
                'layers' : [15],
                'max_epoch' : 2
            }, max_features=2, repeats=2)
          
feature_selector.run()


