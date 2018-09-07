'''
Created on Jul 2, 2018

@author: nishilab
'''

from action_menu import ActionMenu

class BaseEvaluatorActionMenu(ActionMenu):
    
        
    def _get_menu_options(self):
        return ['Display predicted features',
                'Display selected evaluators',
                'Plot target vs predicted',
                'Plot errors',
                'Plot absolute errors',
                'Plot relative errors',
                'Display absolute mean error',
                'Display relative mean error']
        
    def _perform_action(self, action, command):
        
        feature_pos = command.feature
        evals = command.evals
        scaled = command.scaled
        tail = command.tail
        
        if action == 1:
            print(self._actor.predicted_features_str())
        elif action == 2:
            print(self._actor.evaluators_str())
        elif action == 3:
            self._actor.plot_target_vs_predicted(feature_pos, evals=evals, scaled=scaled, tail=tail)
        elif action == 4:
            self._actor.plot_errors(feature_pos, evals=evals, scaled=scaled)
        elif action == 5:
            self._actor.plot_absolute_errors(feature_pos, evals=evals, scaled=scaled)
        elif action == 6:
            self._actor.plot_relative_errors(feature_pos, evals=evals, scaled=scaled)
        elif action == 7:
            print(self._actor.absolute_mean_error_str(feature_pos, scaled=scaled, evals=evals))
        elif action == 8:
            print(self._actor.relative_mean_error_str(feature_pos, scaled=scaled, evals=evals))
        else:
            self._handle_action(action, feature_pos, scaled, tail, evals)
            
    def _handle_action(self, action, feature_pos, scaled, tail, evals):
        raise NotImplementedError("Child classes must implement this method")    
    