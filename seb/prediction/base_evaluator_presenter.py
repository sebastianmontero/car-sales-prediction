'''
Created on Jun 15, 2018

@author: nishilab
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    
    def _plot_target_vs(self, months, real, vs, ylabel, title):
        vs['Real'] = real
        self._plot_by_month_new_process(months, vs, ylabel, title)
    
    def _plot_by_month_new_process(self, months, series, ylabel, title, yfix=False):
        self._run_in_new_process(target=self._plot_by_month, args=(months, series, ylabel, title, yfix))
        
    def _plot_by_month(self, months, series, ylabel, title, yfix=False):
        
        FONT_SIZE = 14
        LINE_WIDTH = 2.5
        min_val = None
        max_val = None
        num_months = len(months)
        
        plt.rc('xtick', labelsize=FONT_SIZE)
        plt.rc('ytick', labelsize=FONT_SIZE)
        plt.rc('legend', fontsize=FONT_SIZE)
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) 
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        i = 0
        for label, obj in series.items():
            vals = obj
            plt_type = 'line'
            x_shift = 0
            if isinstance(obj, dict):
                vals = obj['values']
                plt_type = obj.get('type', 'line')
                x_shift = obj.get('x_shift', 0)
            
            tmin=min(vals)
            tmax=max(vals)
            if min_val is None or tmin < min_val:
                min_val = tmin
            if max_val is None or tmax > max_val:
                max_val = tmax
            range_end = int(num_months - x_shift)
            xvals = range(range_end - len(vals), range_end)
            if plt_type == 'bar':
                plt.bar(xvals, vals, label=(label))
            elif plt_type == 'line':
                plt.plot(xvals, vals, label=(label), linewidth=LINE_WIDTH)
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
    
    def _plot_errors_new_process(self, months, absolute, relative, ylabel_absolute, ylabel_relative, title):
        self._run_in_new_process(target=self._plot_errors, args=(months, absolute, relative, ylabel_absolute, ylabel_relative, title))
            
    def _plot_errors(self, months, absolute, relative, ylabel_absolute, ylabel_relative, title):
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
    
    def plot_target_vs_predicted(self, feature_pos=0, scaled=False, tail=False, evals_pos=[], prediction_indexes=[[0]]):
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        
        predictions = {}
        evals = self.evals(evals_pos)
        prediction_indexes = self._normalize_prediction_indexes(evals, prediction_indexes)
        min_start_month_pos, max_end_month_pos = self._get_month_range(evals, prediction_indexes, tail)
        for pos, evl in enumerate(evals):
            for pi in prediction_indexes[pos]:
                eo = evl['obj']
                predictions[self._evaluator_name(evl['name'], pi)] = {
                    'values': eo.get_predictions(feature_pos, scaled, pi),
                    'x_shift': max_end_month_pos - eo.end_window_pos(pi)
                }

         
        months = ev.get_months(max_end_month_pos, max_end_month_pos - min_start_month_pos)
        self._plot_target_vs(months, ev.get_target_by_start_end(feature_name, min_start_month_pos, max_end_month_pos, scaled=scaled), 
                                                   predictions, 
                                                   formatted_feature_name, 'Real vs Predicted ' + formatted_feature_name) 
    
    def _get_month_range(self, evals, prediction_indexes, tail=False):
        
        min_start_month_pos = None
        max_end_month_pos = None
        for pos, evl in enumerate(evals):
            for pi in prediction_indexes[pos]:
                eo = evl['obj']
                start_month_pos = eo.start_window_pos(pi)
                end_month_pos = eo.end_window_pos(pi)
                if min_start_month_pos is None or start_month_pos < min_start_month_pos:
                    min_start_month_pos = start_month_pos
                if max_end_month_pos is None or end_month_pos > max_end_month_pos:
                    max_end_month_pos = end_month_pos
        
        if tail:
            min_start_month_pos = 0
        return min_start_month_pos, max_end_month_pos
    
    def _normalize_prediction_indexes(self, evals, prediction_indexes):
        complement = np.zeros([len(evals) - len(prediction_indexes),1]).tolist()
        return prediction_indexes + complement
        
    def _evaluator_name(self, name, prediction_index):
        return '{}-P{}'.format(name, int(prediction_index))
        
    def plot_errors(self, feature_pos=0, scaled=False, evals=[], prediction_index=0):
        ev = self.eval(evals)
        ev_name = ev['name']
        ev = ev['obj']
        feature_name = ev.get_predicted_var_name(feature_pos)
        months = ev.get_months(prediction_index=prediction_index)
        absolute = ev.absolute_error(feature_pos, scaled, prediction_index)
        relative = ev.relative_error(feature_pos, scaled, prediction_index)
        title = '{} Target vs Prediction Errors [{}]'.format(self._generate_feature_name(feature_name, scaled), self._evaluator_name(ev_name, prediction_index))
        self._plot_errors_new_process(months, absolute, relative, 'Absolute Error', 'Relative Error', title)
        
    def plot_absolute_errors(self, feature_pos=0, scaled=False, evals=[], prediction_indexes=[[0]]):
        self.plot_errors_by_type('absolute', feature_pos, scaled, evals, prediction_indexes)
    
    def plot_relative_errors(self, feature_pos=0, scaled=False, evals=[], prediction_indexes=[[0]]):
        self.plot_errors_by_type('relative', feature_pos, scaled, evals, prediction_indexes)
            
    def plot_errors_by_type(self, type_='absolute',  feature_pos=0, scaled=False, evals_pos=[], prediction_indexes=[[0]]):
        props = {
            'absolute' : ('Absolute', 'absolute_error'),
            'relative' : ('Relative', 'relative_error')
        }
        
        name, fn =  props[type_]
        
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        errors = {}
        evals = self.evals(evals_pos)
        prediction_indexes = self._normalize_prediction_indexes(evals, prediction_indexes)
        for pos, evl in enumerate(evals):
            for pi in prediction_indexes[pos]:
                errors[self._evaluator_name(evl['name'], pi)] = {
                    'values': getattr(evl['obj'],fn)(feature_pos, scaled, pi),
                    'type':'o'
                }
            
        title = '{} Target vs Prediction {} Errors'.format(self._generate_feature_name(feature_name, scaled), name)
        min_start_month_pos, max_end_month_pos = self._get_month_range(evals, prediction_indexes, tail=False) 
        months = ev.get_months(max_end_month_pos, max_end_month_pos - min_start_month_pos)
        self._plot_by_month_new_process(months, errors, name + ' Error', title)
        
        
    def absolute_mean_error_str(self, feature_pos=0, scaled=False, evals=[], prediction_indexes=[[0]]):
        return self._mean_error_str(self._absolute_mean_error_str, self._total_absolute_mean_error_str, feature_pos, scaled, evals, prediction_indexes)
    
    def _mean_error_str(self, fn, fn_total, feature_pos=0, scaled=False, evals_pos=[], prediction_indexes=[[0]]):
        evals = self.evals(evals_pos)
        str_ = ''
        prediction_indexes = self._normalize_prediction_indexes(evals, prediction_indexes)
        for pos,ev in enumerate(evals):
            pis = prediction_indexes[pos]
            for pi in pis: 
                str_ += fn(ev, feature_pos, scaled, pi) + '\n'
            if len(pis) > 1:
                str_ += '\t' + fn_total(ev, feature_pos, scaled, pis) + '\n'
        return str_
    
    def _absolute_mean_error_str(self, eval_, feature_pos=0, scaled=False, prediction_index=0):
        return self.__absolute_mean_error_str(eval_['obj'].absolute_mean_error_single(feature_pos, scaled, prediction_index), self._evaluator_name(eval_['name'], prediction_index), feature_pos, scaled)
    
    def _total_absolute_mean_error_str(self, eval_, feature_pos=0, scaled=False, prediction_indexes=[]):
        return self.__absolute_mean_error_str(eval_['obj'].absolute_mean_error(feature_pos, scaled, prediction_indexes), eval_['name'], feature_pos, scaled)
    
    def __absolute_mean_error_str(self, error, eval_name, feature_pos=0, scaled=False):
        return '[{}] {} absolute mean error: {:.2f}'.format(eval_name, self.generate_feature_name(feature_pos, scaled=scaled), error)
    
    def relative_mean_error_str(self, feature_pos=0, scaled=False, evals=[], prediction_indexes=[[0]]):
        return self._mean_error_str(self._relative_mean_error_str,self._total_relative_mean_error_str, feature_pos, scaled, evals, prediction_indexes)
    
    def _relative_mean_error_str(self, eval_, feature_pos=0, scaled=False, prediction_index=0):
        return self.__relative_mean_error_str(eval_['obj'].relative_mean_error_single(feature_pos, scaled, prediction_index), self._evaluator_name(eval_['name'], prediction_index), feature_pos, scaled)
    
    def _total_relative_mean_error_str(self, eval_, feature_pos=0, scaled=False, prediction_indexes=[]):
        return self.__relative_mean_error_str(eval_['obj'].relative_mean_error(feature_pos, scaled, prediction_indexes), eval_['name'], feature_pos, scaled)
    
    def __relative_mean_error_str(self, error, eval_name, feature_pos=0, scaled=False):
        return '[{}] {} relative mean error: {:.2f}%'.format(eval_name, self.generate_feature_name(feature_pos, scaled=scaled), error)
    
    
    def predicted_features_str(self):
        ev = self.eval_obj(0)
        features = ev.predicted_features
        str_ = ''
        for i, feature in enumerate(features):
            str_ += '[{}] {} \n'.format(i, self.format_name(feature))
        return str_
    
    def evaluators_str(self):
        str_ = ''
        for i, ev in enumerate(self._evaluators):
            str_ += '[{}] {} \n'.format(i, ev['name'])
        return str_