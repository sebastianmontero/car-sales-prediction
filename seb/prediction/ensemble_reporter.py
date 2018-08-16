from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from storage_manager import StorageManager, StorageManagerType
from utils import Utils
from ensemble_config import EnsembleConfig
from ensemble_evaluator import EnsembleEvaluator
from db_manager import DBManager



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
    
    
    def get_ensemble_evaluator(self):
        
        if self._ensemble_evaluator is None:
            pickle = self._ensemble_eval_sm.get_pickle(self.run_path)
            
            if pickle is None or self._overwrite:
                self._ensemble_evaluator = EnsembleEvaluator(self._get_evaluators())
                self.store_noise_variance_dataset('test')
                self._ensemble_eval_sm.pickle(self._ensemble_evaluator, self.run_path, self._ensemble_evaluator.real_absolute_mean_error())
            else:
                self._ensemble_evaluator = self._ensemble_eval_sm.unpickle(pickle)
                
        return self._ensemble_evaluator
    
    def store_noise_variance_dataset(self, sufix):
        dataset = self.get_ensemble_evaluator().get_noise_variance_dataset()
        dataset.to_sql(name='month_manufacturing_confidence_index',con=DBManager.get_engine(), if_exists='replace', index=False)
        
        
    def _get_evaluators(self):
        evaluators_errors = self._eval_sm.get_objects_errors(self._run_path, recursive=True, sorted_=True, max_=self._num_networks) 
        return [evaluator_error['obj'] for evaluator_error in evaluators_errors]
            
    @classmethod
    def find_ensemble_runs(cls, path):
        return Utils.search_paths(path, EnsembleConfig.BASE_DIR_PREFIX + '*', recursive=True, sort=True)


