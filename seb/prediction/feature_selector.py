from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import datetime
import ray
from ray.tune import grid_search, run_experiments, register_trainable
from ray.tune.async_hyperband import AsyncHyperBandScheduler
from ray.tune.hyperband import HyperBandScheduler 

from model_trainable import ModelTrainable
from feature_selector_reporter import FeatureSelectorReporter


class FeatureSelector():
     
    
    FEATURES = ['consumer_confidence_index',
                'exchange_rate',
                'interest_rate',
                'manufacturing_confidence_index',
                'economic_activity_index',
                'energy_price_index_roc_prev_month',
                'energy_price_index_roc_start_year',
                'inflation_index_roc_prev_month',
                'inflation_index_roc_start_year']
    
    '''FEATURES = ['consumer_confidence_index',
                'energy_price_index_roc_prev_month']'''
    
    def __init__(self, config, max_features=9, repeats = 3):        
        assert (max_features <= len(self.FEATURES)), "max_features {} should be less than the number of possible features {}".format(max_features, len(self.FEATURES))
        self._repeats = repeats 
        self._max_features = max_features
        config['store_window'] = False
        self._config = config
        self._current_selected_features = []
        self._reporter = FeatureSelectorReporter(base_path=ModelTrainable.BASE_PATH)
        self._config['save_path'] = self._reporter.run_path
        self._ray_results_dir = os.path.join(os.path.expanduser('~'), 'ray_results', self._reporter.get_experiments_base_dir())
    
    def _feature_search_space(self, current_selected_features):
        free_features = [feature for feature in self.FEATURES if feature not in current_selected_features]
        space = []
        for feature in free_features:
            space_item = [feature]
            space_item.extend(current_selected_features)
            space.append(space_item)
        
        return space
    
    
    def run(self):
        for num_features in range(1, self._max_features + 1):
            config = self._config.copy()
            config['included_features'] = grid_search(self._feature_search_space(self._current_selected_features))
            experiment_name = self._reporter.get_experiment_name(num_features) 
            
            run_experiments({
                experiment_name : {
                    'run': 'car_sales_prediction_trainable',
                    'trial_resources': {'cpu': 8, 'gpu': 1},
                    #'stop': {'neg_mean_loss': 0, 'training_iteration': 200},
                    'stop': {'training_iteration': 350},
                    'config' : config,
                    'repeat':self._repeats,
                    'local_dir': self._ray_results_dir
                }
            })
            best_config = self._reporter.get_best_config(experiment_name)
            self._current_selected_features = best_config['obj']['included_features']
        
        print('Finished feature search!')
        
        self._reporter.print_best_config()
        self._reporter.print_best_configs()
    

ray.init()
register_trainable('car_sales_prediction_trainable', ModelTrainable)

feature_selector = FeatureSelector({
                'keep_prob' : 1,
                'layers' : [15],
                'max_epoch' : 2,
                'window_size': 37
            }, max_features=9, repeats=3)
          
feature_selector.run()


