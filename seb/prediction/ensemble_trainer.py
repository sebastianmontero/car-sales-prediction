from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import ray
from ray.tune import grid_search, run_experiments, register_trainable

from model_trainable import ModelTrainable
from ensemble_config import EnsembleConfig


class EnsembleTrainer():
     
    
    def __init__(self, config, repeats, description=''):        
        self._repeats = repeats 
        self._config = config
        config['store_window'] = False
        self._config = config
        self._ensemble_config = EnsembleConfig(description=description, base_path=ModelTrainable.BASE_PATH)
        self._ray_results_dir = os.path.join(os.path.expanduser('~'), 'ray_results', self._ensemble_config.get_ensemble_base_dir())
         
    
    def run(self):
        
        experiment_name = self._ensemble_config.get_ensemble_base_dir() 
            
        run_experiments({
            experiment_name : {
                'run': 'car_sales_prediction_trainable',
                'trial_resources': {'cpu': 8, 'gpu': 1},
                'stop': {'training_iteration': 70},
                'config' : self._config,
                'repeat':self._repeats,
                'local_dir': self._ray_results_dir
            }
        })  
        print('Finished ensemble training!')
    

ray.init()
register_trainable('car_sales_prediction_trainable', ModelTrainable)


ensemble_trainer = EnsembleTrainer({
                'line_id': 13,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37, 40]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 1,
                'store_window' : False,
                'included_features' : ['energy_price_index_roc_prev_month'],
                'predicted_features':['sales'],
                'multi_month_prediction':False,
                'num_steps': 70
            }, repeats=4, description='model_1n_r4')

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 102,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37, 40]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 3,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                      'manufacturing_confidence_index'],
                'predicted_features':['sales'],
                'multi_month_prediction':False,
                'num_steps': 70
            }, repeats=4, description='platform_3n_r4')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 13,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37, 40]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 3,
                'store_window' : False,
                'included_features' : ['energy_price_index_roc_prev_month'],
                'predicted_features':['sales'],
                'multi_month_prediction':True,
                'num_steps': 70
            }, repeats=4, description='model_3n_mp_r4')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 201,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37, 40]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 2,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                   'consumer_confidence_index'],
                'predicted_features':['sales'],
                'multi_month_prediction':False,
                'num_steps': 70
            }, repeats=4, description='nationwide_2n_r4')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 201,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37, 40]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 3,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                   'consumer_confidence_index'],
                'predicted_features':['sales'],
                'multi_month_prediction':True,
                'num_steps': 70
            }, repeats=4, description='nationwide_3n_mp_r4')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 13,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37, 40]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 1,
                'store_window' : False,
                'included_features' : ['energy_price_index_roc_prev_month'],
                'num_steps': 50
            }, repeats=3, description='model_sf_ifp_2m')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 13,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 2,
                'store_window' : False,
                'included_features' : ['energy_price_index_roc_prev_month'],
                'predicted_features':['sales'],
                'num_steps': 50
            }, repeats=4, description='model_2n')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 201,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 3,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                   'consumer_confidence_index'],
                'predicted_features':['sales'],
                'num_steps': 50
            }, repeats=3, description='nationwide_3n')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 201,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([35, 40, 45, 50, 55, 60]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 1,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                   'consumer_confidence_index'],
                'num_steps': 40
            }, repeats=3, description='nationwide_sf_ifp_3m')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 102,
                'keep_prob' : grid_search([0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 1,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                      'manufacturing_confidence_index'],
                'num_steps': grid_search([12, 24])
            }, repeats=4, description='platform_26_37')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 102,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([15, 20, 25, 30]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 1,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                      'manufacturing_confidence_index'],
                'num_steps': 12
            }, repeats=4, description='platform_x5')'''
'''ensemble_trainer = EnsembleTrainer({
                'line_id': 201,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37]),
                'max_epoch' : 2,
                'train_months': 36,
                'prediction_size': 2,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                   'consumer_confidence_index'],
                'num_steps': grid_search([12, 24])
            }, repeats=3, description='nationwide_2n')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 102,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([12, 15, 18, 21, 24]),
                'max_epoch' : 2,
                'window_size': 37,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                      'manufacturing_confidence_index'],
                'num_steps': grid_search([12, 24])
            }, repeats=3, description='platform_small')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 102,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37]),
                'max_epoch' : 2,
                'window_size': 37,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                      'manufacturing_confidence_index'],
                'num_steps': grid_search([12, 24])
            }, repeats=3, description='platform')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 13,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([12, 15, 18, 21, 24]),
                'max_epoch' : 2,
                'window_size': 37,
                'store_window' : False,
                'included_features' : ['energy_price_index_roc_prev_month'],
                'num_steps': grid_search([12, 24])
            }, repeats=3, description='model_small')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 13,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37]),
                'max_epoch' : 2,
                'window_size': 37,
                'store_window' : False,
                'included_features' : ['energy_price_index_roc_prev_month'],
                'num_steps': grid_search([12, 24])
            }, repeats=3, description='model')'''

'''ensemble_trainer = EnsembleTrainer({
                'line_id': 201,
                'keep_prob' : grid_search([0.8, 0.9, 1.0]),
                'layer_0' : grid_search([26, 29, 31, 34, 37]),
                'max_epoch' : 2,
                'window_size': 25,
                'store_window' : False,
                'included_features' : ['inflation_index_roc_prev_month',
                                   'consumer_confidence_index'],
                'num_steps': grid_search([12, 24])
            }, repeats=3, description='nationwide')'''
          
ensemble_trainer.run()


