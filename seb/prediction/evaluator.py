'''
Created on Jun 15, 2018

@author: nishilab
'''

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
import glob
import re
from enum import Enum

from utils import Utils


class PickleAction(Enum):
    KEEP = 'keep'
    OVERWRITE = 'overwrite'
    BEST = 'best'
    NOTHING = 'nothing'

class Evaluator(object):
    
    PICKLE_FILE_NAME_PREFIX = 'evaluator-pickle-'

    def __init__(self, reader, predictions, end_window_pos):
        self._reader = reader
        self._unscaled_predictions = self._get_real_sales_from_predictions(predictions)
        self._predictions = np.reshape(predictions, [-1])
        self._end_window_pos = end_window_pos
        self._window_length = len(predictions)
        sns.set()
        
    def _plot_target_vs_predicted(self, real, predictions, ylabel, title):
        months = self._get_months()
        plt.plot(real, label=('Real'))
        plt.plot(predictions, label=('Predicted'))
        plt.ylabel(ylabel)
        plt.xticks(range(len(months)), months, rotation='vertical')
        plt.title(title)
        plt.legend()
        plt.show()
        
    def _plot_errors(self, absolute, relative, ylabel_absolute, ylabel_relative, title):
        months = self._get_months()
        color = 'tab:blue'
        fig, ax1 = plt.subplots()
        ax1.set_xticks(range(len(months)))
        ax1.set_xticklabels(months)
        ax1.tick_params(axis='x', labelrotation=90)
        
        ax1.set_ylabel(ylabel_absolute, color=color)
        ax1.plot(absolute, 'o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        color = 'tab:green'
        ax2 = ax1.twinx()
        ax2.plot(relative, 'o', color=color)
        ax2.set_ylabel(ylabel_relative, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
            
        plt.suptitle(title)
        plt.show()
        
        
    def _calculate_absolute_error_by_pos(self, targets, predictions, pos):
        return math.fabs(targets[pos]- predictions[pos])
    
    def _calculate_absolute_error(self, targets, predictions):
        return np.abs(targets - predictions)
    
    def _calculate_relative_error(self, targets, predictions):
        return np.abs((targets - predictions)/targets) * 100
    
    def _calculate_absolute_mean_error(self, targets, predictions):
        return np.mean(self._calculate_absolute_error(targets, predictions))
    
    def _calculate_relative_mean_error(self, targets, predictions):
        return np.mean(self._calculate_relative_error(targets, predictions))
        
    def _get_real_sales_from_predictions(self, predictions):
        unscaled = np.reshape(self._reader.unscale_sales(predictions), [-1])
        return [round(max(prediction,0)) for prediction in unscaled]
        
    def _get_target_sales(self, scaled=False):
        return self._get_data(scaled)['sales'].values

    def _calculate_and_plot_errors(self, scaled=False):
        targets = self._get_target_sales(scaled)
        predictions = self._get_predictions(scaled)
        absolute = self._calculate_absolute_error(targets, predictions)
        relative = self._calculate_relative_error(targets, predictions)
        title = 'Target vs Prediction Errors' + (' (Scaled)' if scaled else '')
        self._plot_errors(absolute, relative, 'Absolute Error', 'Relative Error', title)
    
    def _get_months(self):
        return self._get_data()['month_id'].values
    
    def _get_data(self, scaled=False):
        return self._reader.get_data(self._end_window_pos, self._window_length, scaled)
    
    def _get_predictions(self, scaled=False):
        return self._predictions if scaled else self._unscaled_predictions
    
    def plot_real_target_vs_predicted(self):
        self._plot_target_vs_predicted(self._get_target_sales(), self._get_predictions(), 'Sales', 'Real vs Predicted Sales')
        
    def plot_scaled_target_vs_predicted(self):
        self._plot_target_vs_predicted(self._get_target_sales(scaled=True), self._get_predictions(scaled=True), 'Sales', 'Scaled Real vs Predicted Sales')
    
    def plot_real_errors(self):
        return self._calculate_and_plot_errors()
    
    def plot_scaled_errors(self):
        return self._calculate_and_plot_errors(scaled=True)
    
    def real_absolute_mean_error(self):
        return self._calculate_absolute_mean_error(self._get_target_sales(), self._get_predictions())
    
    def scaled_absolute_mean_error(self):
        return self._calculate_absolute_mean_error(self._get_target_sales(scaled=True), self._get_predictions(scaled=True))
    
    def real_relative_mean_error(self):
        return self._calculate_relative_mean_error(self._get_target_sales(), self._get_predictions())
    
    def scaled_relative_mean_error(self):
        return self._calculate_relative_mean_error(self._get_target_sales(scaled=True), self._get_predictions(scaled=True))
    
    def _get_absolute_error_by_pos(self, pos, scaled=False):
        return self._calculate_absolute_error_by_pos(self._get_target_sales(scaled), self._get_predictions(scaled), pos)
    
    def real_absolute_error_by_pos(self, pos):
        return self._get_absolute_error_by_pos(pos)
    
    def scaled_absolute_error_by_pos(self, pos):
        return self._get_absolute_error_by_pos(pos, True)
    
    def pickle(self, path, error, pickle_action=PickleAction.OVERWRITE):
        Evaluator.pickle_obj(self, path, error, pickle_action)
    
    @classmethod
    def _get_pickle_file_path(cls, path, error):
        return os.path.join(path, cls.PICKLE_FILE_NAME_PREFIX + str(error) + '.bin')
    
    @classmethod
    def unpickle(cls, path):
        if os.path.isdir(path):
            path = cls._get_pickle_file_path()
        
        with open(path, mode='rb') as file:
            return pickle.load(file)
    
    @classmethod
    def _remove_pickles(cls, path):
        Utils.remove_files_from_dir(path, [cls.PICKLE_FILE_NAME_PREFIX])
        
    @classmethod
    def _get_pickle_file_name(cls, pickle):
        return os.path.split(pickle)[1]
    
    @classmethod
    def _sort_pickles_by_error(cls, pickles):
        pickles.sort(key=cls._get_pickle_file_name)
    
    @classmethod
    def _get_best_pickle(cls, path=None, pickles=None):
        assert (path or pickles), "Either path or pickles must be specified"
        
        if path:
            pickles = cls.get_pickles(path)
        return None if len(pickles) == 0 else pickles[0]
    
    @classmethod
    def _get_best_pickle_error(cls, path=None, pickles=None):
        best = cls._get_best_pickle(path, pickles)
        return cls._get_error_from_pickle(best) if best else None
    
    @classmethod    
    def _get_error_from_pickle(cls, pickle):
        name = cls._get_pickle_file_name(pickle)
        return float(name[len(cls.PICKLE_FILE_NAME_PREFIX):-4])
        
    @classmethod
    def _filter_pickles(cls, pickles, filter_, start_pos=0):
        return list(filter(lambda pickle: re.search(filter_, pickle[start_pos:]), pickles))
        
    @classmethod
    def get_pickles(cls, path, filter_=None, recursive=False, sorted_=True):
        
        path_wild_card = '**' if recursive else ''
        path = os.path.join(path, path_wild_card, cls.PICKLE_FILE_NAME_PREFIX + '*.bin')
        pickles = glob.glob(path, recursive=recursive)
        
        if filter_:
            pickles = cls._filter_pickles(pickles, filter_)
        
        if sorted_:
            cls._sort_pickles_by_error(pickles)
        return pickles
        
    @classmethod
    def pickle_obj(cls, obj, path, error, pickle_action=PickleAction.OVERWRITE):
        
        if PickleAction.BEST:
            best_error = cls._get_best_pickle_error(path)
            pickle_action = PickleAction.OVERWRITE if best_error is None or error < best_error else PickleAction.NOTHING
        
        if pickle_action == PickleAction.OVERWRITE:
            cls._remove_pickles(path)
            
        if pickle_action != PickleAction.NOTHING:
            pickle_file = cls._get_pickle_file_path(path, error)
            os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
            with open(pickle_file, mode='wb') as file:
                pickle.dump(obj, file)
    