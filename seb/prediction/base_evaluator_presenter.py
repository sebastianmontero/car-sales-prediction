'''
Created on Jun 15, 2018

@author: nishilab
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Process

from utils import Utils

class BaseEvaluatorPresenter(object):

    def __init__(self, evaluators):
        sns.set()
        self._evaluators = np.array(evaluators)
        
    def eval_obj(self, pos=0):
        return self._evaluators[pos]['obj']
    
    def eval_name(self, pos=0):
        return self._evaluators[pos]['name']
    
    def evals(self, evals):
        if len(evals) == 0:
            return self._evaluators
        return self._evaluators[evals]
    
    def eval(self, evals):
        pos = 0 if len(evals) == 0 else evals[0]
        return self._evaluators[pos]
        
    def _run_in_new_process(self, target, args=()):
        p = Process(target=target, args=args)
        p.start()
    
    def _plot_target_vs_predicted_new_process(self, real, predictions, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs, args=(real, predictions, ylabel, title))
    
    def _plot_target_vs(self, real, vs, ylabel, title):
        vs['Real'] = real
        self._plot_by_month('Real', vs, ylabel, title)
    
    def _plot_by_month_new_process(self, ref_key, series, ylabel, title, yfix=False):
        self._run_in_new_process(target=self._plot_by_month, args=(ref_key, series, ylabel, title, yfix))
        
    def _plot_by_month(self, ref_key, series, ylabel, title, yfix=False):
        ref_vals = series[ref_key]
        if isinstance(ref_vals, dict):
            ref_vals = ref_vals['values']
        months = self.eval_obj().get_months(len(ref_vals))
        num_months = len(months)
        
        min_val = None
        max_val = None
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        i = 0
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
            
            xvals = range(num_months - len(vals), num_months)
            if plt_type == 'bar':
                plt.bar(xvals, vals, label=(label))
            elif plt_type == 'line':
                plt.plot(xvals, vals, label=(label))
            else:
                plt.plot(xvals, vals, plt_type, label=(label), color=colors[i])
            i += 1
        if yfix:
            plt.ylim([min_val-0.5*(max_val-min_val), max_val+0.5*(max_val-min_val)])
        
        plt.ylabel(ylabel)
        plt.xticks(range(num_months), months, rotation='vertical')
        plt.title(title)
        plt.legend()
        plt.show()
    
    def _plot_errors_new_process(self, absolute, relative, ylabel_absolute, ylabel_relative, title):
        self._run_in_new_process(target=self._plot_errors, args=(absolute, relative, ylabel_absolute, ylabel_relative, title))
            
    def _plot_errors(self, absolute, relative, ylabel_absolute, ylabel_relative, title):
        months = self.eval_obj().get_months()
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
    
    def plot_target_vs_predicted(self, feature_pos=0, scaled=False, tail=False, evals=[]):
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        
        predictions = {}
        for evl in self.evals(evals):
            predictions[evl['name']] = evl['obj'].get_predictions(feature_pos, scaled)
        
        self._plot_target_vs_predicted_new_process(ev.get_target(feature_name, scaled=scaled, length=ev.get_target_data_length(tail)), 
                                                   predictions, 
                                                   formatted_feature_name, 'Real vs Predicted ' + formatted_feature_name) 
    
    def plot_errors(self, feature_pos=0, scaled=False, evals=[]):
        ev = self.eval(evals)
        ev_name = ev['name']
        ev = ev['obj']
        feature_name = ev.get_predicted_var_name(feature_pos)
        absolute = ev.absolute_error(feature_pos, scaled)
        relative = ev.relative_error(feature_pos, scaled)
        title = '{} Target vs Prediction Errors [{}]'.format(self._generate_feature_name(feature_name, scaled), ev_name)
        self._plot_errors_new_process(absolute, relative, 'Absolute Error', 'Relative Error', title)
        
    def plot_absolute_errors(self, feature_pos=0, scaled=False, evals=[]):
        self.plot_errors_by_type('absolute', feature_pos, scaled, evals)
    
    def plot_relative_errors(self, feature_pos=0, scaled=False, evals=[]):
        self.plot_errors_by_type('relative', feature_pos, scaled, evals)
            
    def plot_errors_by_type(self, type_='absolute',  feature_pos=0, scaled=False, evals=[]):
        props = {
            'absolute' : ('Absolute', 'absolute_error'),
            'relative' : ('Relative', 'relative_error')
        }
        
        name, fn =  props[type_]
        
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        errors = {}
        for evl in self.evals(evals):
            errors[evl['name']] = {
                'values': getattr(evl['obj'],fn)(feature_pos, scaled),
                'type':'o'
            }
            
        title = '{} Target vs Prediction {} Errors'.format(self._generate_feature_name(feature_name, scaled), name)
        self._plot_by_month_new_process(self.eval_name(), errors, name + ' Error', title)
        
        
    def absolute_mean_error_str(self, feature_pos=0, scaled=False, evals=[]):
        return self._mean_error_str(self._absolute_mean_error_str, feature_pos, scaled, evals)
    
    def _mean_error_str(self, fn, feature_pos=0, scaled=False, evals=[]):
        evs = self.evals(evals)
        
        str_ = ''
        for ev in evs:
            str_ += self.fn(ev, feature_pos, scaled) + '\n'
        return str_
    
    def _absolute_mean_error_str(self, eval_, feature_pos=0, scaled=False):
        return '[{}] {} absolute mean error: {:.2f}'.format(eval_['name'], self.generate_feature_name(feature_pos, scaled=scaled), eval_['obj'].absolute_mean_error(feature_pos, scaled))
    
    def relative_mean_error_str(self, feature_pos=0, scaled=False, evals=[]):
        return self._mean_error_str(self._relative_mean_error_str, feature_pos, scaled, evals)
    
    def _relative_mean_error_str(self, eval_, feature_pos=0, scaled=False):
        return '[{}] {} relative mean error: {:.2f}%'.format(eval_['name'], self.generate_feature_name(feature_pos, scaled=scaled), eval_['obj'].relative_mean_error(feature_pos, scaled))
    
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