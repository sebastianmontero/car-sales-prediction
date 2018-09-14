from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import re
import traceback

from storage_manager import StorageManager, StorageManagerType
from utils import Utils
from ensemble_config import EnsembleConfig
from ensemble_evaluator import EnsembleEvaluator
from db_manager import DBManager
from ensemble_evolver import EnsembleEvolver


class InvalidIFP(Exception):
    pass

class EnsembleReporter():
    
    def __init__(self, run_path, num_networks=None, overwrite=False):
                
        self._ensemble_config = EnsembleConfig(run_path=run_path)
        self._run_path = run_path
        self._num_networks = num_networks
        self._eval_sm = StorageManager.get_storage_manager(StorageManagerType.EVALUATOR)
        self._ensemble_eval_sm = StorageManager.get_storage_manager(StorageManagerType.ENSEMBLE_EVALUATOR)
        self._ensemble_evaluator = None
        self._overwrite = overwrite
        
    @property
    def run_path(self):
        return self._run_path
    
    
    def get_ensemble_evaluator(self, operator='mean', find_best_ensemble=False):
        
        if self._ensemble_evaluator is None:
            pickle = self._ensemble_eval_sm.get_pickle(self.run_path)
            
            if pickle is None or self._overwrite:
                self._ensemble_evaluator = EnsembleEvaluator(self._get_evaluators(), operator, find_best_ensemble)
                self._ensemble_eval_sm.pickle(self._ensemble_evaluator, self.run_path, self._ensemble_evaluator.absolute_mean_error())
            else:
                self._ensemble_evaluator = self._ensemble_eval_sm.unpickle(pickle)
                
        return self._ensemble_evaluator
    
    def evolve_ensemble(self, config):  
        evaluator = self.get_ensemble_evaluator()
        evolver = EnsembleEvolver(config, evaluator)
        evolver.evolve()
        
        
    def _get_evaluators(self):
        evaluators_errors = self._eval_sm.get_objects_errors(self._run_path, recursive=1, sorted_=True, max_=self._num_networks) 
        return [evaluator_error['obj'] for evaluator_error in evaluators_errors]
            
    @classmethod
    def find_ensemble_runs(cls, path, filter_=None, exclude_filter=None, sort=True):
        return Utils.search_paths(path, EnsembleConfig.BASE_DIR_PREFIX + '*', recursive=False, sort=sort, filter_=filter_, exclude_filter=exclude_filter)
    
    @classmethod
    def find_base_ensemble_runs(cls, path):
        name = os.path.basename(path)
        ifp_data = cls._process_ifp_run(name)
        if not ifp_data:
            return []
        
        base_dir = os.path.dirname(path)
        
        base_name = ifp_data['base_name']
        ensembles = cls.find_ensemble_runs(base_dir, filter_=base_name, exclude_filter=name)
        
        base_predictions = ifp_data['prediction'] - 1
        if len(ensembles) != (base_predictions):
            raise InvalidIFP('Invalid number of base ensemble runs') 
        
        for i, ensemble in enumerate(ensembles):
            if not os.path.basename(ensemble).startswith('{}{}m'.format(base_name, i + 1)):
                raise InvalidIFP('Invalid base ensemble runs')
            
        return ensembles
    
    @classmethod
    def _process_ifp_run(cls, name):
        
        if name.startswith(EnsembleConfig.BASE_DIR_PREFIX):
            match = re.search(EnsembleConfig.IFP_INDICATOR, name)
            if match:
                try:
                    return {
                        'base_name': name[0:match.end()], 
                        'prediction': int(name[match.end() : name.index('m', match.end())])
                    }
                except:
                    traceback.print_exc()
                    raise InvalidIFP('Invalid Input Feature Prediction Name')
        return False
    
'''reporter = EnsembleReporter(run_path='/home/nishilab/Documents/python/model-storage/ensemble-run-model-20180815225523217235',
                            overwrite=True)
reporter.evolve_ensemble({
        'num_generations': 20
    })'''

