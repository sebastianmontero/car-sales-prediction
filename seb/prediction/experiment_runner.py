from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import ray
from ray.tune import grid_search, run_experiments, register_trainable
from ray.tune.async_hyperband import AsyncHyperBandScheduler
from ray.tune.hyperband import HyperBandScheduler 

from model_trainable import ModelTrainable
from ray.tune.variant_generator import grid_search
                
ray.init()

register_trainable('car_sales_prediction_trainable', ModelTrainable)

'''run_experiments({
        'experiment1' : {
                'run': 'car_sales_prediction_trainable',
                'trial_resources': {'cpu': 8, 'gpu': 1},
                #'stop': {'neg_mean_loss': -2, 'training_iteration': 10},
                'stop': {'training_iteration': 100},
                'config' : {
                    'keep_prob' : grid_search(np.arange(0.2, 1.1, 0.1).tolist()),
                    'max_epoch' : 1
                },
                'checkpoint_freq': 1
            }
    }, scheduler=AsyncHyperBandScheduler(time_attr='training_iteration', 
                                         reward_attr='neg_mean_loss',
                                         grace_period=3,
                                         reduction_factor=3,
                                         brackets=3))'''
'''run_experiments({
        'network_structure' : {
                'run': 'car_sales_prediction_trainable',
                'trial_resources': {'cpu': 8, 'gpu': 1},
                'stop': {'neg_mean_loss': -2, 'training_iteration': 1},
                'config' : {
                    'keep_prob' : grid_search(np.linspace(0.4, 1., 4).tolist()),
                    'layer_0' : grid_search([30, 70, 110]),
                    'layer_1' : grid_search([None, 30, 70, 110]),
                    'max_epoch' : 1,
                    'included_features' : ['consumer_confidence_index',
                                           'exchange_rate',
                                           'interest_rate',
                                           'manufacturing_confidence_index',
                                           'economic_activity_index',
                                           'energy_price_index_roc_prev_month',
                                           'energy_price_index_roc_start_year',
                                           'inflation_index_roc_prev_month',
                                           'inflation_index_roc_start_year']
                }
            }
    })'''

''', scheduler=HyperBandScheduler(time_attr='training_iteration', 
                                         reward_attr='neg_mean_loss',
                                         max_t=100))'''
        
'''run_experiments({
    'best_feature' : {
            'run': 'car_sales_prediction_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 1},
            'stop': {'neg_mean_loss': -2, 'training_iteration': 1},
            'config' : {
                'keep_prob' : 1,
                'layers' : [15],
                'max_epoch' : 1,
                'included_features' : grid_search([['consumer_confidence_index'],
                                       ['exchange_rate'],
                                       ['interest_rate'],
                                       ['manufacturing_confidence_index'],
                                       ['economic_activity_index'],
                                       ['energy_price_index_roc_prev_month'],
                                       ['energy_price_index_roc_start_year'],
                                       ['inflation_index_roc_prev_month'],
                                       ['inflation_index_roc_start_year']])
            }
        }
})'''

print('Experiment start')


run_experiments({
    'network_structure_complement' : {
        'run': 'car_sales_prediction_trainable',
        'trial_resources': {'cpu': 8, 'gpu': 1},
        'stop': {'training_iteration': 50},
        'config' : {
            'line_id': 201,
            'keep_prob' : grid_search([0.5, 0.75, 1.0]),
            'layer_0' : 15,
            'layer_1' : 15,
            'max_epoch' : 2,
            'window_size': 37,
            'store_window' : False,
            'included_features' : ['inflation_index_roc_prev_month',
                                   'consumer_confidence_index',
                                   'energy_price_index_roc_prev_month']
        },
        'repeat':1
    }
})

'''run_experiments({
    'network_structure' : {
        'run': 'car_sales_prediction_trainable',
        'trial_resources': {'cpu': 8, 'gpu': 1},
        'stop': {'training_iteration': 50},
        'config' : {
            'line_id': 201,
            'keep_prob' : grid_search([0.5, 0.75, 1.0]),
            'layer_0' : grid_search([5, 15, 25]),
            'layer_1' : grid_search([None, 5, 15, 25]),
            'max_epoch' : 2,
            'window_size': 37,
            'store_window' : False,
            'included_features' : ['inflation_index_roc_prev_month',
                                   'consumer_confidence_index',
                                   'energy_price_index_roc_prev_month']
        },
        'repeat':3
    }
})'''


'''run_experiments({
    'num_steps_coarse_nationwide' : {
            'run': 'car_sales_prediction_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 1},
            'stop': {'training_iteration': 60},
            'config' : {
                'line_id': 201,
                'keep_prob' : 1,
                'layers' : [15],
                'max_epoch' : 2,
                'window_size': 37,
                'store_window' : False,
                'included_features' : ['energy_price_index_roc_prev_month'],
                'num_steps': grid_search([12,24,36])
            },
            'repeat':3
        }
})'''

print('Experiment end')
          
        
        