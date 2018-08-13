from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import datetime

import pprint

from storage_manager import StorageManager, StorageManagerType
from utils import Utils
from ensemble_config import EnsembleConfig



class EnsembleReporter():
    
    def __init__(self, run_path, num_networks=None):
                
        self._ensembleConfig = EnsembleConfig(run_path)
        self._run_path = run_path
        self._num_networks = num_networks
        
        self._eval_sm = StorageManager.get_storage_manager(StorageManagerType.EVALUATOR)
        self._best_config = None
        self._best_configs = None
        self._pprint = pprint.PrettyPrinter()
        self._configs_map = {}
    
    @property
    def best_configs(self):
        if self._best_configs is None:
            self._process_run()
            best_configs = []
            
            for _, configs in self._configs_map.items():
                best_configs.append(configs[0])
            best_configs.sort(key=lambda x: len(x['obj']['included_features']))
            self._best_configs = best_configs
        return self._best_configs
    
    @property
    def best_config(self):
        if self._best_config is None:
            best = None
            for config in self.best_configs:
                if best is None or config['avg_error'] < best['avg_error']:
                    best = config
            self._best_config = best
        return self._best_config
    
    @property
    def run_path(self):
        return self._run_path
    
    def get_best_config(self, experiment_name):
        return self._process_experiment(experiment_name)[0]
    
    def _get_num_features_from_path(self, experiment_path):
        _, experiment_dir = os.path.split(experiment_path)
        return int(experiment_dir[len(self.EXPERIMENT_NAME_PREFIX):-21])
    
    def _process_run(self):
        for experiment_name in os.listdir(self.run_path):
            self._process_experiment(experiment_name)
            
    def _find_experiment_name(self, num_features):
        name = None
        prefix = self._get_experiment_name_prefix(num_features)
        for experiment_name in os.listdir(self.run_path):
            if experiment_name.startswith(prefix):
                name = experiment_name
                break
        
        return name
    
    def get_experiment_configs(self, num_features):
        if num_features not in self._configs_map:
            experiment_name = self._find_experiment_name(num_features)
            if experiment_name is None:
                raise ValueError('No experiment for {} number of features exist'.format(num_features))
            self._process_experiment(experiment_name)
        return self._configs_map[num_features]
        
    def _get_evaluators(self):
        return self._eval_sm.get_objects_errors(self._run_path, recursive=True, sorted_=True, max=self._num_networks)
    
    def _find_best_config(self, configs):
        best =  None
        for config in configs:
            if best is None or config['avg_error'] < best['avg_error']:
                best = config
        
        return best
    
    def get_experiment_name(self, num_features):
        return '{}-{}'.format(self._get_experiment_name_prefix(num_features), datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
    
    def _get_experiment_name_prefix(self, num_features):
        return '{}{}'.format(self.EXPERIMENT_NAME_PREFIX, str(num_features))
    
    
    def _get_experiment_path(self, experiment_name):
        return os.path.join(self.run_path, experiment_name)
    
    def _generate_experiments_base_dir(self):
        return '{}{}'.format(self.BASE_DIR_PREFIX, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
    
    def get_experiments_base_dir(self):
        _,dir = os.path.split(self._run_path)
        return dir
    
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
        
    def print_experiment_configs(self, num_features):
        configs = self.get_experiment_configs(num_features)
        print()
        print("Configurations for {} number of features:".format(num_features))
        print()
        self._pprint.pprint(configs)
        print()
        
    @classmethod
    def find_feature_selector_runs(cls, path):
        return Utils.search_paths(path, cls.BASE_DIR_PREFIX + '*', recursive=True, sort=True)


