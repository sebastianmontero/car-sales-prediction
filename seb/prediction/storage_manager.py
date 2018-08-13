'''
Created on Jun 15, 2018

@author: nishilab
'''

import pickle
import os
import re
from enum import Enum

from utils import Utils

class StorageManagerType(Enum):
    EVALUATOR = 'evaluator-pickle-'
    CONFIG = 'config-pickle-'

class PickleAction(Enum):
    KEEP = 'keep'
    OVERWRITE = 'overwrite'
    BEST = 'best'
    NOTHING = 'nothing'

class StorageManager(object):

    def __init__(self, file_name_prefix):
        self._file_name_prefix = file_name_prefix
    
    def _get_pickle_file_path(self, path, error):
        return os.path.join(path, self._file_name_prefix + str(error) + '.bin')
    
    def unpickle(self, path):
        if os.path.isdir(path):
            path = self._get_best_pickle(path)
            print('Unpickle:', path)
        
        with open(path, mode='rb') as file:
            return pickle.load(file)
    
    def _remove_pickles(self, path):
        Utils.remove_files_from_dir(path, [self._file_name_prefix])
        
    def _get_pickle_file_name(self, pickle):
        return os.path.split(pickle)[1]
    
    def _sort_pickles_by_error(self, pickles):
        pickles.sort(key=self._get_pickle_file_name)
    
    def _get_best_pickle(self, path=None, pickles=None):
        assert (path or pickles), "Either path or pickles must be specified"
        
        if path:
            pickles = self.get_pickles(path)
        return None if len(pickles) == 0 else pickles[0]
    
    def _get_best_pickle_error(self, path=None, pickles=None):
        best = self._get_best_pickle(path, pickles)
        return self._get_error_from_pickle(best) if best else None
        
    def _get_error_from_pickle(self, pickle):
        name = self._get_pickle_file_name(pickle)
        return float(name[len(self._file_name_prefix):-4])
            
    def get_pickles(self, path, filter_=None, recursive=False, sorted_=True, exclude_filter=None):
        
        pickles = Utils.search_paths(path, self._file_name_prefix + '*.bin', recursive, filter_=filter_, exclude_filter=exclude_filter)
        
        if sorted_:
            self._sort_pickles_by_error(pickles)
        return pickles
    
    def get_objects_errors(self, path, filter_=None, recursive=False, sorted_=True, max_= None):
        
        pickles = self.get_pickles(path, filter_, recursive, sorted_)
        
        if max_:
            pickles = pickles[:max]
        
        objs = []
        
        for pickle in pickles:
            objs.append({
                'obj': self.unpickle(pickle),
                'error': self._get_error_from_pickle(pickle)
            })
        return objs
        
    def pickle(self, obj, path, error, pickle_action=PickleAction.OVERWRITE):

        if PickleAction.BEST:
            best_error = self._get_best_pickle_error(path)
            pickle_action = PickleAction.OVERWRITE if best_error is None or error < best_error else PickleAction.NOTHING
        
        if pickle_action == PickleAction.OVERWRITE:
            self._remove_pickles(path)
            
        if pickle_action != PickleAction.NOTHING:
            pickle_file = self._get_pickle_file_path(path, error)
            os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
            with open(pickle_file, mode='wb') as file:
                pickle.dump(obj, file)
                
    
    @classmethod
    def get_storage_manager(cls, type):
        return StorageManager(type.value)