from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import datetime


class EnsembleConfig():
    
    BASE_DIR_PREFIX = 'ensemble-run-'
    IFP_INDICATOR = '_ifp_'
    
    def __init__(self, description='', base_path = None, run_path = None):
        assert(base_path or run_path), "base_path or run_path must be specified"
        self._description = description
        if base_path:
            run_path = os.path.join(base_path, self._generate_ensemble_base_dir())        
        self._run_path = run_path
    
    @property
    def run_path(self):
        return self._run_path
    
    def _generate_ensemble_base_dir(self):
        return '{}{}-{}'.format(self.BASE_DIR_PREFIX, self._description, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
    
    def get_ensemble_base_dir(self):
        _,dir = os.path.split(self._run_path)
        return dir
    


