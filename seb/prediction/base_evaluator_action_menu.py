'''
Created on Jul 2, 2018

@author: nishilab
'''

from action_menu import ActionMenu

class BaseEvaluatorActionMenu(ActionMenu):
    
        
    def _get_menu_options(self):
        return ['Display predicted features',
                'Plot target vs predicted real',
                'Plot target vs predicted real with tail',
                'Plot target vs predicted scaled',
                'Plot target vs predicted scaled with tail',
                'Plot real errors',
                'Plot scaled errors',
                'Show real absolute mean error',
                'Show scaled absolute mean error',
                'Show real relative mean error',
                'Show scaled relative mean error']
        
    def _perform_action(self, action, params):
        feature_pos = 0
        
        if len(params) > 1:
            feature_pos = int(params[1])
        
        if action == 1:
            self._display_predicted_vars()
        elif action == 2:
            self._actor.plot_target_vs_predicted(feature_pos)
        elif action == 3:
            self._actor.plot_target_vs_predicted(feature_pos, tail=True)
        elif action == 4:
            self._actor.plot_target_vs_predicted(feature_pos, scaled=True)
        elif action == 5:
            self._actor.plot_target_vs_predicted(feature_pos, scaled=True, tail=True)
        elif action == 6:
            self._actor.plot_errors(feature_pos)
        elif action == 7:
            self._actor.plot_errors(feature_pos, scaled=True)
        elif action == 8:
            print('{} absolute mean error: {:.2f}'.format(self._actor.generate_feature_name(feature_pos, scaled=False), self._actor.absolute_mean_error(feature_pos)))
        elif action == 9:
            print('{} absolute mean error: {:.2f}'.format(self._actor.generate_feature_name(feature_pos, scaled=True), self._actor.absolute_mean_error(feature_pos, scaled=True)))
        elif action == 10:
            print('{} relative mean error: {:.2f}%'.format(self._actor.generate_feature_name(feature_pos, scaled=False), self._actor.relative_mean_error(feature_pos)))
        elif action == 11:
            print('{} relative mean error: {:.2f}%'.format(self._actor.generate_feature_name(feature_pos, scaled=True), self._actor.relative_mean_error(feature_pos, scaled=True)))
        else:
            self._handle_action(action, feature_pos, params)
            
    def _handle_action(self, action, feature_pos, params):
        raise NotImplementedError("Child classes must implement this method")
    
    def _display_predicted_vars(self):
        features = self._actor.predicted_vars
        
        for i, feature in enumerate(features):
            print('[{}] {}'.format(i, self._actor.format_name(feature)))
        
    