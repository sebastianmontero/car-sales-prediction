'''
Created on Jun 15, 2018

@author: nishilab
'''

import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Process

from utils import Utils

class BaseEvaluatorPresenter(object):

    def __init__(self, evaluators):
        sns.set()
        self._evaluators = evaluators
        
    def eval_obj(self, pos):
        return self._evaluators[pos]['obj']
        
    def _run_in_new_process(self, target, args=()):
        p = Process(target=target, args=args)
        p.start()
    
    def _plot_target_vs_predicted_new_process(self, real, predictions, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_predicted, args=(real, predictions, ylabel, title))
    
    def _plot_target_vs(self, real, vs, ylabel, title):
        vs['Real'] = real
        self._plot_by_month('Real', vs, ylabel, title)
        
    def _plot_by_month(self, ref_key, series, ylabel, title, yfix=False):
        ref_vals = series[ref_key]
        if isinstance(ref_vals, dict):
            ref_vals = ref_vals['values']
        months = self.eval_obj(0).get_months(len(ref_vals))
        num_months = len(months)
        
        min_val = None
        max_val = None
        for label, obj in series.items():
            vals = obj
            plt_type = 'line'
            if isinstance(obj, dict):
                vals = obj['values']
                plt_type = obj.get('type', 'line')
            
            tmin=min(vals)
            tmax=max(vals)
            if min_val is None or tmin < min_val:
                min_val = tmin
            if max_val is None or tmax > max_val:
                max_val = tmax
            
            if plt_type == 'bar':
                plt.bar(range(num_months - len(vals), num_months), vals, label=(label))
            else:
                plt.plot(range(num_months - len(vals), num_months), vals, label=(label))
        
        if yfix:
            plt.ylim([min_val-0.5*(max_val-min_val), max_val+0.5*(max_val-min_val)])
        
        plt.ylabel(ylabel)
        plt.xticks(range(num_months), months, rotation='vertical')
        plt.title(title)
        plt.legend()
        plt.show()
        
    def _plot_target_vs_predicted(self, real, predictions, ylabel, title):
        self._plot_target_vs(real, {'Predicted' : predictions}, ylabel, title)
    
    def _plot_errors_new_process(self, absolute, relative, ylabel_absolute, ylabel_relative, title):
        self._run_in_new_process(target=self._plot_errors, args=(absolute, relative, ylabel_absolute, ylabel_relative, title))
            
    def _plot_errors(self, absolute, relative, ylabel_absolute, ylabel_relative, title):
        months = self.eval_obj(0).get_months()
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
    
    
    def _generate_feature_name(self, feature_name, scaled=None):
        name = ''
        if scaled is not None:
            name += 'Scaled ' if scaled else 'Real '
        name +=  self.format_name(feature_name)
        return name.strip()
    
    def generate_feature_name(self, feature_pos, scaled=None):
        return self._generate_feature_name(self.eval_obj(0).get_predicted_var_name(feature_pos), scaled)
    
    def format_name(self, name):
        ns = name.split('_')
        fname = ''
        
        for n in ns:
            fname += n[0].capitalize() + n[1:] + ' '
            
        return fname  
    
    def plot_target_vs_predicted(self, feature_pos=0, scaled=False, tail=False):
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
         
        self._plot_target_vs_predicted_new_process(ev.get_target(feature_name, scaled=scaled, length=ev.get_target_data_length(tail)), 
                                                   ev.get_predictions(feature_pos, scaled), 
                                                   formatted_feature_name, 'Real vs Predicted ' + formatted_feature_name) 
    
    def plot_errors(self, feature_pos=0, scaled=False):
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        targets = ev.get_target(feature_name, scaled=scaled)
        predictions = ev.get_predictions(feature_pos, scaled=scaled)
        absolute = ev.calculate_absolute_error(targets, predictions)
        relative = ev.calculate_relative_error(targets, predictions)
        title = self._generate_feature_name(feature_name, scaled) + ' Target vs Prediction Errors'
        self._plot_errors_new_process(absolute, relative, 'Absolute Error', 'Relative Error', title)
        
        
    def absolute_mean_error_str(self, feature_pos=0, scaled=False):
        ev = self.eval_obj(0)
        return '{} absolute mean error: {:.2f}'.format(self.generate_feature_name(feature_pos, scaled=scaled), ev.absolute_mean_error(feature_pos, scaled))
    
    def relative_mean_error_str(self, feature_pos=0, scaled=False):
        ev = self.eval_obj(0)
        return '{} relative mean error: {:.2f}%'.format(self.generate_feature_name(feature_pos, scaled=scaled), ev.relative_mean_error(feature_pos, scaled))
        
    
    def predicted_vars_str(self):
        ev = self.eval_obj(0)
        features = ev.predicted_vars
        str_ = ''
        for i, feature in enumerate(features):
            str_ += '[{}] {} \n'.format(i, self.format_name(feature))
        return str_
    
    def evaluators_str(self):
        str_ = ''
        for i, ev in enumerate(self._evaluators):
            str_ += '[{}] {} \n'.format(i, ev['name'])
        return str_