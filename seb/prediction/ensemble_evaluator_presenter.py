'''
Created on Jun 15, 2018

@author: nishilab
'''

from utils import Utils
from base_evaluator_presenter import BaseEvaluatorPresenter

class EnsembleEvaluatorPresenter(BaseEvaluatorPresenter):

    def _plot_target_vs_ensemble_best_new_process(self, real, ensemble, best, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_ensemble_best, args=(real, ensemble, best, ylabel, title))
        
    def _plot_target_vs_ensemble_best(self, real, ensemble, best, ylabel, title):
        self._plot_target_vs(real,{'Ensemble':ensemble, 'Best Network': best},ylabel, title)
        
    def plot_target_vs_ensemble_best(self, feature_pos=0, scaled=False, tail=False):
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        self._plot_target_vs_ensemble_best_new_process(ev.get_target(feature_name, scaled=scaled,length=ev.get_target_data_length(tail)), 
                                                   ev.get_predictions(feature_pos, scaled), 
                                                   ev.best_network.get_predictions(feature_pos, scaled), 
                                                   formatted_feature_name, 
                                                   'Target vs Ensemble and Best Network ' + formatted_feature_name)
        
    
    def _plot_target_vs_ensemble_min_max_new_process(self, real, ensemble, min_, max_, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_ensemble_min_max, args=(real, ensemble, min_, max_, ylabel, title))
        
    def _plot_target_vs_ensemble_min_max(self, real, ensemble, min_, max_, ylabel, title):
        self._plot_target_vs(real,{'Ensemble':ensemble, 'Min': min_, 'Max': max_},ylabel, title)
        
    def plot_target_vs_ensemble_min_max(self, feature_pos=0, scaled=False, tail=False):
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        self._plot_target_vs_ensemble_min_max_new_process(ev.get_target(feature_name, scaled=scaled, length=ev.get_target_data_length(tail)), 
                                                      ev.get_predictions(feature_pos, scaled), 
                                                      ev.get_min(feature_pos, scaled), 
                                                      ev.get_max(feature_pos, scaled) , 
                                                      formatted_feature_name, 
                                                      'Target vs Ensemble, Min and Max ' + formatted_feature_name)
        
    def _plot_target_vs_mean_interval_new_process(self, real, mean, lower, upper, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_mean_interval, args=(real, mean, lower, upper, ylabel, title))
        
    def _plot_target_vs_mean_interval(self, real, mean, lower, upper, ylabel, title):
        self._plot_target_vs(real,{'Ensemble Mean':mean, 'Lower Limit': lower, 'Upper Limit': upper}, ylabel, title)
        
    def plot_target_vs_mean_interval(self, feature_pos=0, scaled=False, tail=False):
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        self._plot_target_vs_mean_interval_new_process(ev.get_target(feature_name, scaled=scaled,length=ev.get_target_data_length(tail)), 
                                                       ev.get_predictions(feature_pos, scaled), 
                                                       ev.get_lower(feature_pos, scaled), 
                                                       ev.get_upper(feature_pos, scaled) , 
                                                       formatted_feature_name, 
                                                       'Target vs Ensemble Mean and Interval ' + formatted_feature_name)
        
    
    def _plot_variance_errors_new_process(self, model_variance, noise_variance, ylabel, title):
        self._run_in_new_process(target=self._plot_variance_errors, args=(model_variance, noise_variance, ylabel, title))
        
    def _plot_variance_errors(self, model_variance, noise_variance, ylabel, title):
        self._plot_by_month('Model Variance',{'Model Variance':model_variance, 'Noise Variance': noise_variance}, ylabel, title)
        
    def plot_variance_errors(self, feature_pos=0):
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled=None)
        self._plot_variance_errors_new_process(ev.get_model_variance(feature_pos), 
                                               ev.get_noise_variance(feature_pos), 
                                               formatted_feature_name + ' Variance', 'Model and Noise {} Variance'.format(formatted_feature_name))
        
    
    def _plot_std_new_process(self, std, ylabel, title):
        self._run_in_new_process(target=self._plot_std, args=(std, ylabel, title))
        
    def _plot_std(self, std, ylabel, title):
        self._plot_by_month('Standard Deviation',{'Standard Deviation':{'values': std, 'type': 'bar'}}, ylabel, title, yfix=True)
        
    def plot_std(self, feature_pos=0, scaled=False):
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled=scaled)
        title = formatted_feature_name + ' Standard Deviation'
        self._plot_std_new_process(ev.get_std(feature_pos, scaled=scaled),
                                   title, 
                                   title)
    
    def _plot_min_max_range_new_process(self, mm_range, ylabel, title):
        self._run_in_new_process(target=self._plot_min_max_range, args=(mm_range, ylabel, title))
        
    def _plot_min_max_range(self, mm_range, ylabel, title):
        self._plot_by_month('Min Max Range',{'Min Max Range':{'values': mm_range, 'type': 'bar'}}, ylabel, title, yfix=True)
        
    def plot_min_max_range(self, feature_pos=0, scaled=False):
        ev = self.eval_obj(0)
        feature_name = ev.get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled=scaled)
        title = formatted_feature_name + ' Min Max Range'
        self._plot_min_max_range_new_process(ev.get_min_max_range(feature_pos, scaled=scaled),
                                             title, 
                                             title)
        
    def absolute_mean_error_str(self, feature_pos=0, scaled=False):
        ev = self.eval_obj(0)
        ame_str = super(EnsembleEvaluatorPresenter, self).absolute_mean_error_str(feature_pos, scaled)
        return '{} Best Network: {:.2f}'.format(ame_str, ev.best_network.absolute_mean_error(feature_pos, scaled))

    
    def relative_mean_error_str(self, feature_pos=0, scaled=False):
        ev = self.eval_obj(0)
        rme_str = super(EnsembleEvaluatorPresenter, self).relative_mean_error_str(feature_pos, scaled)
        return '{} Best Network: {:.2f}%'.format(rme_str, ev.best_network.relative_mean_error(feature_pos, scaled))
    