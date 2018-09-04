from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from ray.tune import Trainable, TrainingResult
from ray.tune.async_hyperband import AsyncHyperBandScheduler
from ray.tune.hyperband import HyperBandScheduler 

from model_trainer import ModelTrainer

        
class ModelTrainable(Trainable):
    
    BASE_PATH = '/home/nishilab/Documents/python/model-storage'
    
    def _setup(self):
        self.timesteps = 0
        experiment_path, trial_dir =  os.path.split(self.logdir)
        _, experiment_dir =  os.path.split(experiment_path)
        base_path = self.config['save_path'] if 'save_path' in self.config else ModelTrainable.BASE_PATH
        trial_dir = self._remove_invalid_path_chars(trial_dir)
        self.config['save_path'] = os.path.join(base_path, experiment_dir, trial_dir)
        self.model_trainer = ModelTrainer(self.config)
        
    def _train(self):
        evaluator = self.model_trainer.train()
        self.timesteps += 1
        return TrainingResult(timesteps_total=self.timesteps, timesteps_this_iter=1, training_iteration=self.timesteps, mean_loss=evaluator.absolute_mean_error())
    
    def _save(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        print('Save: ', self.logdir, checkpoint_dir)
        return os.path.join(checkpoint_dir, 'ray')      
    
    def _restore(self, path):
        print('Restore: ', self.logdir, path)
        
    def _remove_invalid_path_chars(self, path):
        invalid_chars = "[]'"
        for c in invalid_chars:
            path = path.replace(c, '')
            
        return path
          
        
        