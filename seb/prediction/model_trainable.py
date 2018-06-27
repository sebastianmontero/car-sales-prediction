from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import ray
from ray.tune import grid_search, run_experiments, register_trainable, Trainable, TrainingResult
from ray.tune.async_hyperband import AsyncHyperBandScheduler
from ray.tune.hyperband import HyperBandScheduler 

from model_trainer import ModelTrainer

        
class ModelTrainable(Trainable):
    
    def _setup(self):
        self.timesteps = 0
        self.config['save_path'] = self.logdir
        self.model_trainer = ModelTrainer(self.config)
        
    def _train(self):
        evaluator = self.model_trainer.train()
        self.timesteps += 1
        return TrainingResult(timesteps_total=self.timesteps, timesteps_this_iter=1, training_iteration=self.timesteps, mean_loss=evaluator.real_absolute_mean_error())
    def _save(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        print('Save: ', self.logdir, checkpoint_dir)
        return os.path.join(checkpoint_dir, 'ray')      
    def _restore(self, path):
        print('Restore: ', self.logdir, path)
                

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
run_experiments({
        'experiment1' : {
                'run': 'car_sales_prediction_trainable',
                'trial_resources': {'cpu': 8, 'gpu': 1},
                'stop': {'neg_mean_loss': -2, 'training_iteration': 100},
                'config' : {
                    'keep_prob' : grid_search(np.arange(0.2, 1.1, 0.1).tolist()),
                    'max_epoch' : 5
                },
                'checkpoint_freq': 1
            }
    })

''', scheduler=HyperBandScheduler(time_attr='training_iteration', 
                                         reward_attr='neg_mean_loss',
                                         max_t=100))'''
        
        
        
        