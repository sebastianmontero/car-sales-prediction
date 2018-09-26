'''
Created on Jun 15, 2018

@author: nishilab
'''

import numpy as np

from utils import Utils
from base_evaluator import BaseEvaluator
from scipy.stats import t
from sklearn.cluster import MeanShift

class InvalidEnsembleWeights(Exception):
    pass

class EnsembleEvaluator(BaseEvaluator):

    def __init__(self, evaluators, operator='mean', find_best_ensemble=False):
        BaseEvaluator.__init__(self)
        
        self._num_networks = len(evaluators)
        assert (self._num_networks > 1), "There must be at least two evaluators in the ensemble"
        
        best_network = evaluators[0]
        self._quantile = 0.975 #0.95 confidence interval 
        self._best_network = best_network
        self._end_window_pos = best_network._end_window_pos
        self._window_length = best_network.window_length
        self._reader = best_network.reader
        self._operator = operator
        self._weights = np.ones((self._num_networks), dtype=np.int)
        self._evals_predictions = self._generate_predictions_array(evaluators)    
        self._mean = None
        self._mean_u = None
        self._mode = None
        self._mode_u = None
        self._median = None
        self._median_u = None
        self._model_variance = None
        self._noise_variance = None
        self._std = None
        self._std_u = None
        self._min = None
        self._max = None
        self._min_u = None
        self._max_u = None
        self._lower = None
        self._lower_u = None
        self._upper = None
        self._upper_u = None
        
        if find_best_ensemble:
            self._find_best_ensemble()
        else:
            self._process_evaluators()
        
    @property
    def window_length(self):
        return self._window_length
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._set_weights(weights)
        self._process_evaluators()
        
    def _set_weights(self, weights):
        self._weights = np.array(list(map(lambda x: 0 if x < 0 else x, weights)))
        
    def _find_best_ensemble(self):
        candidate_pos = 0
        rme = None
        weights = np.zeros((self._num_networks), dtype=np.int)
        while candidate_pos is not None:
            weights[candidate_pos] = 1
            print('Chosen: ', candidate_pos)
            candidate_pos = None
            if rme:
                print('Current best relative mean error: {}'.format(rme))
            for pos in range(self._num_networks):
                if weights[pos] == 0:
                    weights[pos] = 1
                    erme = self.test_ensemble(weights, 90)
                    if rme is None or erme < rme:
                        candidate_pos = pos
                        rme = erme
                    weights[pos] = 0
        self.weights = weights
    
    def test_ensemble(self, weights, subset=100):
        self._set_weights(weights)
        predictions, weights = self._filter_zero_networks(self._evals_predictions, self.weights)
        self._calculate_ensemble_prediction(predictions, weights)
        return self.relative_mean_error(subset=subset)
        
    def _process_evaluators(self):
        predictions, weights = self._filter_zero_networks(self._evals_predictions, self.weights)
        self._calculate_ensemble_prediction(predictions, weights)
        self._model_variance = self._calculate_model_variance(predictions)
        self._noise_variance = self._calculate_noise_variance(self.get_predicted_targets(scaled=True),self._mean, self._model_variance)
        self._std = self._calculate_std(predictions)
        self._min = self._get_min(predictions)
        self._max = self._get_max(predictions)
        self._lower, self._upper = self._calculate_interval(self._mean, self._std)
        self._std_u = self._unscale_features(self._std, round_=False)
        self._min_u = self._unscale_features(self._min)
        self._max_u = self._unscale_features(self._max)
        self._lower_u = self._unscale_features(self._lower)
        self._upper_u = self._unscale_features(self._upper)
        
    def _filter_zero_networks(self, predictions, weights):
        non_zero_pos = np.nonzero(weights)[0]
        if len(non_zero_pos) == 0:
            raise InvalidEnsembleWeights("No non zero weights")
        return predictions[:,non_zero_pos], weights[non_zero_pos]
        
        
    def _calculate_ensemble_prediction(self, predictions, weights):
        
        if self._operator == 'mean':
            self._mean = self._calculate_mean(predictions, weights)
            self._mean_u = self._unscale_features(self._mean)
        elif self._operator == 'median':
            self._median = self._calculate_median(predictions)
            self._median_u = self._unscale_features(self._median)
        elif self._operator == 'mode':
            self._mode = self._calculate_mode(predictions)
            self._mode_u = self._unscale_features(self._mode)
        
    @property
    def best_network(self):
        return self._best_network
    
    def _generate_predictions_array(self, evaluators):
        predictions = []
        for e in evaluators:
            predictions.append(np.reshape(e.predictions(scaled=True), [-1, 1, e.num_predicted_vars]))
        
        return np.concatenate(predictions, axis=1)
    
    def _calculate_mean(self, predictions, weights):
        return np.average(predictions, axis=1, weights=weights)
    
    def _calculate_median(self, predictions):
        return np.median(predictions, axis=1)
    
    def _calculate_mode(self, predictions):
        mode = []
        prev_month_predictions = None
        for month_predictions in predictions:
            ms = MeanShift()
            try:
                ms.fit(month_predictions)
            except:
                ms.set_params(bandwidth=0.75)
                ms.fit(month_predictions)
            if prev_month_predictions is not None:
                cc = np.array(ms.cluster_centers_)
                distances = np.linalg.norm(cc - prev_month_predictions, axis=1)
                sel_predictions = cc[np.argmin(distances)]
            else:
                sel_predictions = np.array(ms.cluster_centers_[0])
            prev_month_predictions = sel_predictions
            mode.append(sel_predictions)
        
        return mode
    
    def _calculate_model_variance(self, predictions):
        return np.var(predictions, axis=1, ddof=1)
    
    def _calculate_noise_variance(self, targets, mean, model_variance):
        return [np.maximum((t - m) ** 2 - v, np.zeros(t.shape)) for t, m, v in zip(targets, mean, model_variance)]
    
    def _calculate_std(self, predictions):
        return np.std(predictions, axis=1, ddof=1)
    
    def _calculate_interval(self, mean, std):
        range_ = std * t.ppf(self._quantile, self._num_networks - 1)
        return mean - range_, mean + range_ 
    
    def _get_min(self, predictions):
        return np.amin(predictions, axis=1)
    
    def _get_max(self, predictions):
        return np.amax(predictions, axis=1)
    
    def mean(self, scaled=False):
        return self._mean if scaled else self._mean_u
    
    def median(self, scaled=False):
        return self._median if scaled else self._median_u
    
    def mode(self, scaled=False):
        return self._mode if scaled else self._mode_u
    
    def std(self, scaled=False):
        return self._std if scaled else self._std_u
    
    def min(self, scaled=False):
        return self._min if scaled else self._min_u
    
    def max(self, scaled=False):
        return self._max if scaled else self._max_u
    
    def min_max_range(self, scaled=False):
        return self.max(scaled) - self.min(scaled)
    
    def lower(self, scaled=False):
        return self._lower if scaled else self._lower_u
    
    def upper(self, scaled=False):
        return self._upper if scaled else self._upper_u
    
    #Has to be defined so that its compatible with evaluator, this method is used in BaseEvaluators
    def predictions(self, scaled=False):
        fn = {
                'mean': self.mean,
                'median' : self.median,
                'mode' : self.mode
              }
        return fn[self._operator](scaled)
    
    def get_predictions(self, feature_pos=0, scaled=False, prediction_index=0):
        return self._get_feature_values(self.predictions(scaled), self.feature_index(feature_pos, prediction_index))
    
    def get_std(self, feature_pos=0, scaled=False, prediction_index=0):
        return self._get_feature_values(self.std(scaled), self.feature_index(feature_pos, prediction_index))
    
    def get_min(self, feature_pos=0, scaled=False, prediction_index=0):
        return self._get_feature_values(self.min(scaled), self.feature_index(feature_pos, prediction_index))
    
    def get_max(self, feature_pos=0, scaled=False, prediction_index=0):
        return self._get_feature_values(self.max(scaled), self.feature_index(feature_pos, prediction_index))
    
    def get_min_max_range(self, feature_pos=0, scaled=False, prediction_index=0):
        return self.get_max(feature_pos, scaled, prediction_index) - self.get_min(feature_pos, scaled, prediction_index)
    
    def get_lower(self, feature_pos=0, scaled=False, prediction_index=0):
        return self._get_feature_values(self.lower(scaled), self.feature_index(feature_pos, prediction_index))
    
    def get_upper(self, feature_pos=0, scaled=False, prediction_index=0):
        return self._get_feature_values(self.upper(scaled), self.feature_index(feature_pos, prediction_index))
    
    def get_model_variance(self, feature_pos=0, prediction_index=0):
        return self._get_feature_values(self._model_variance, self.feature_index(feature_pos, prediction_index))
    
    def get_noise_variance(self, feature_pos=0, prediction_index=0):
        return self._get_feature_values(self._noise_variance, self.feature_index(feature_pos, prediction_index))
    