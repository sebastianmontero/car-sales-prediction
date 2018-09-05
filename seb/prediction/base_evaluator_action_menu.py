'''
Created on Jul 2, 2018

@author: nishilab
'''

from action_menu import ActionMenu

class BaseEvaluatorActionMenu(ActionMenu):
    
        
    def _get_menu_options(self):
        return ['Display predicted features',
                'Display selected evaluators',
                'Plot target vs predicted real',
                'Plot target vs predicted real with tail',
                'Plot target vs predicted scaled',
                'Plot target vs predicted scaled with tail',
                'Plot real errors',
                'Plot scaled errors']
        
    def _perform_action(self, action, command):
        
        feature_pos = command.feature
        evals = command.evals
        
        if action == 1:
            print(self._actor.predicted_vars_str())
        elif action == 2:
            print(self._actor.evaluators_str())
        elif action == 3:
            self._actor.plot_target_vs_predicted(feature_pos)
        elif action == 4:
            self._actor.plot_target_vs_predicted(feature_pos, tail=True)
        elif action == 5:
            self._actor.plot_target_vs_predicted(feature_pos, scaled=True)
        elif action == 6:
            self._actor.plot_target_vs_predicted(feature_pos, scaled=True, tail=True)
        elif action == 7:
            self._actor.plot_errors(feature_pos)
        elif action == 8:
            self._actor.plot_errors(feature_pos, scaled=True)
        else:
            self._handle_action(action, feature_pos, evals)
            
    def _handle_action(self, action, feature_pos, evals):
        raise NotImplementedError("Child classes must implement this method")    
    