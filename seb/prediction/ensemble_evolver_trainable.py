from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from ray.tune import Trainable, TrainingResult

from ensemble_reporter import EnsembleReporter
from ensemble_evolver import EnsembleEvolver
from storage_manager import StorageManager, StorageManagerType, PickleAction
        
class EnsembleEvolverTrainable(Trainable):
    
    BASE_PATH = '/home/nishilab/Documents/python/model-storage'
    
    def _setup(self):
        experiment_path, trial_dir =  os.path.split(self.logdir)
        _, experiment_dir =  os.path.split(experiment_path)
        base_path = self.config['save_path'] if 'save_path' in self.config else EnsembleEvolverTrainable.BASE_PATH
        trial_dir = self._remove_invalid_path_chars(trial_dir)
        self.config['save_path'] = os.path.join(base_path, experiment_dir, trial_dir)
        start_idx = len('ensemble_evolver_trainable') + 1
        end_idx = trial_dir.find('_', start_idx)
        self.trial_number = trial_dir[len('ensemble_evolver_trainable') + 1:end_idx]
        ensemble_reporter = EnsembleReporter(run_path=self.config['run_path'], overwrite=True)
        evaluator = ensemble_reporter.get_ensemble_evaluator()
        self.ensemble_evolver = EnsembleEvolver(self.config, evaluator)
        
    def _train(self):
        gens_per_step = self.config['gens_per_step']
        _, rme = self.ensemble_evolver.evolve_step(gens_per_step)
        generation = self.ensemble_evolver.generation
        return TrainingResult(timesteps_total=generation, timesteps_this_iter=gens_per_step, mean_loss=rme)
    
    def _save(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        print('Save: ', self.logdir, checkpoint_dir)
        return os.path.join(checkpoint_dir, 'ray')      
    
    def _restore(self, path):
        print('Restore: ', self.logdir, path)
        
    def _stop(self):
        best_ensemble = self.ensemble_evolver.get_best_ensemble()
        _, ensemble_dir =  os.path.split(self.config['run_path'])
        ensemble_evaluator_sm = StorageManager.get_storage_manager(StorageManagerType.ENSEMBLE_EVALUATOR)
        evolver_stats_sm = StorageManager.get_storage_manager(StorageManagerType.EVOLVER_STATS)
        file_name_prefix = '{}-GA-{}-'.format(ensemble_dir, self.trial_number)
        rme = best_ensemble.relative_mean_error()
        ensemble_evaluator_sm.pickle(best_ensemble, self.config['save_path'], rme, PickleAction.NOTHING, file_name_prefix)
        evolver_stats_sm.pickle(self.ensemble_evolver.logbook, self.config['save_path'], rme, PickleAction.NOTHING)
        
    def _remove_invalid_path_chars(self, path):
        invalid_chars = "[]'"
        for c in invalid_chars:
            path = path.replace(c, '')
            
        return path
          
        
        