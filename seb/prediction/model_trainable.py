from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import ray
from ray.tune import grid_search, run_experiments, register_trainable, Trainable, TrainingResult 

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
        print('Save: ', self.model_trainer.save_path, checkpoint_dir)
        return os.path.join(checkpoint_dir, 'ray')      
    def _restore(self, path):
        print('Restore: ', self.model_trainer.save_path, path)
                

ray.init()

register_trainable('car_sales_prediction_trainable', ModelTrainable)

run_experiments({
        'experiment1' : {
                'run': 'car_sales_prediction_trainable',
                'trial_resources': {'cpu': 8, 'gpu': 1},
                'stop': {'neg_mean_loss': -2, 'timesteps_total': 3},
                'config' : {
                    'keep_prob' : grid_search(np.arange(0.2, 1.1, 0.1).tolist())
                },
                'checkpoint_freq': 1
            }
    })

        
        
        
        