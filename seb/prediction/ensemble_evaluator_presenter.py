'''
Created on Jun 15, 2018

@author: nishilab
'''

from utils import Utils
from base_evaluator_presenter import BaseEvaluatorPresenter

class EnsembleEvaluatorPresenter(BaseEvaluatorPresenter):
    
    def _plot_target_vs_ensemble_best(self, months, real, ensemble, best, ylabel, title):
        self._plot_target_vs(months, real,{'Ensemble':ensemble, 'Best Network': best},ylabel, title)
        
    def plot_target_vs_ensemble_best(self, feature_pos=0, scaled=False, tail=False, evals=[], prediction_index=0):
        ev = self.eval(evals)
        ev_name = ev['name']
        ev = ev['obj']
        feature_name = ev.get_predicted_var_name(feature_pos)
        months = ev.get_months(prediction_index)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        self._plot_target_vs_ensemble_best(months, ev.get_target(feature_name, scaled=scaled,length=ev.get_target_data_length(tail, prediction_index), prediction_index=prediction_index), 
                                                   ev.get_predictions(feature_pos, scaled, prediction_index=prediction_index), 
                                                   ev.best_network.get_predictions(feature_pos, scaled, prediction_index=prediction_index), 
                                                   formatted_feature_name, 
                                                   'Target vs Ensemble and Best Network {} [{}]'.format(formatted_feature_name, ev_name))
        
        
    def _plot_target_vs_ensemble_min_max(self, months, real, ensemble, min_, max_, ylabel, title):
        self._plot_target_vs(months, real,{'Ensemble':ensemble, 'Min': min_, 'Max': max_},ylabel, title)
        
    def plot_target_vs_ensemble_min_max(self, feature_pos=0, scaled=False, tail=False, evals=[], prediction_index=0):
        ev = self.eval(evals)
        ev_name = ev['name']
        ev = ev['obj']
        feature_name = ev.get_predicted_var_name(feature_pos)
        months = ev.get_months(prediction_index)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        self._plot_target_vs_ensemble_min_max(months, ev.get_target(feature_name, scaled=scaled, length=ev.get_target_data_length(tail, prediction_index), prediction_index=prediction_index), 
                                                      ev.get_predictions(feature_pos, scaled, prediction_index), 
                                                      ev.get_min(feature_pos, scaled, prediction_index), 
                                                      ev.get_max(feature_pos, scaled, prediction_index) , 
                                                      formatted_feature_name, 
                                                      'Target vs Ensemble, Min and Max {} [{}]'.format(formatted_feature_name, ev_name))
            
    def _plot_target_vs_mean_interval(self, months, real, mean, lower, upper, ylabel, title):
        self._plot_target_vs(months, real,{'Ensemble Mean':mean, 'Lower Limit': lower, 'Upper Limit': upper}, ylabel, title)
        
    def plot_target_vs_mean_interval(self, feature_pos=0, scaled=False, tail=False, evals=[], prediction_index=0):
        ev = self.eval(evals)
        ev_name = ev['name']
        ev = ev['obj']
        feature_name = ev.get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        months = ev.get_months(prediction_index)
        self._plot_target_vs_mean_interval(months, ev.get_target(feature_name, scaled=scaled,length=ev.get_target_data_length(tail, prediction_index), prediction_index=prediction_index), 
                                                       ev.get_predictions(feature_pos, scaled, prediction_index), 
                                                       ev.get_lower(feature_pos, scaled, prediction_index), 
                                                       ev.get_upper(feature_pos, scaled, prediction_index) , 
                                                       formatted_feature_name, 
                                                       'Target vs Ensemble Mean and Interval {} [{}]'.format(formatted_feature_name, ev_name))
        
        
    def _plot_variance_errors(self, months, model_variance, noise_variance, ylabel, title):
        self._plot_by_month_new_process(months,{'Model Variance':model_variance, 'Noise Variance': noise_variance}, ylabel, title)
        
    def plot_variance_errors(self, feature_pos=0, evals=[], prediction_index=0):
        ev = self.eval(evals)
        ev_name = ev['name']
        ev = ev['obj']
        feature_name = ev.get_predicted_var_name(feature_pos)
        months = ev.get_months(prediction_index)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled=None)
        self._plot_variance_errors(months, ev.get_model_variance(feature_pos), 
                                               ev.get_noise_variance(feature_pos), 
                                               formatted_feature_name + ' Variance', 'Model and Noise {} Variance [{}]'.format(formatted_feature_name, ev_name))
        
    
    
    def _plot_std(self, months, std, ylabel, title):
        self._plot_by_month_new_process(months,{'Standard Deviation':{'values': std, 'type': 'bar'}}, ylabel, title, yfix=True)
        
    def plot_std(self, feature_pos=0, scaled=False, evals=[], prediction_index=0):
        ev = self.eval(evals)
        ev_name = ev['name']
        ev = ev['obj']
        feature_name = ev.get_predicted_var_name(feature_pos)
        months = ev.get_months(prediction_index)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled=scaled)
        title = '{} Standard Deviation'.format(formatted_feature_name, ev_name)
        self._plot_std(months, ev.get_std(feature_pos, scaled=scaled),
                                   title, 
                                   title)
        
    def _plot_min_max_range(self, months, mm_range, ylabel, title):
        self._plot_by_month_new_process(months,{'Min Max Range':{'values': mm_range, 'type': 'bar'}}, ylabel, title, yfix=True)
        
    def plot_min_max_range(self, feature_pos=0, scaled=False, evals=[], prediction_index=0):
        ev = self.eval(evals)
        ev_name = ev['name']
        ev = ev['obj']
        feature_name = ev.get_predicted_var_name(feature_pos)
        months = ev.get_months(prediction_index)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled=scaled)
        title = '{} Min Max Range [{}]'.format(formatted_feature_name, ev_name)
        self._plot_min_max_range(months, ev.get_min_max_range(feature_pos, scaled=scaled),
                                             title, 
                                             title)
    
    def _absolute_mean_error_str(self, eval_, feature_pos=0, scaled=False):
        ame_str = super(EnsembleEvaluatorPresenter, self)._absolute_mean_error_str(eval_, feature_pos, scaled)
        return '{} Best Network: {:.2f}'.format(ame_str, eval_['obj'].best_network.absolute_mean_error(feature_pos, scaled))
       
    def _relative_mean_error_str(self, eval_, feature_pos=0, scaled=False):
        rme_str = super(EnsembleEvaluatorPresenter, self)._relative_mean_error_str(eval_, feature_pos, scaled)
        return '{} Best Network: {:.2f}%'.format(rme_str, eval_['obj'].best_network.relative_mean_error(feature_pos, scaled))