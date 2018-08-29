'''
Created on Jul 2, 2018

@author: nishilab
'''
from ensemble_reporter import EnsembleReporter
from storage_manager import StorageManager, StorageManagerType
from base_evaluator_action_menu import BaseEvaluatorActionMenu

class EnsembleEvaluatorActionMenu(BaseEvaluatorActionMenu):
    
    def __init__(self, config_sm):
        BaseEvaluatorActionMenu.__init__(self, 'Ensemble Evaluator',StorageManager.get_storage_manager(StorageManagerType.ENSEMBLE_EVALUATOR), config_sm)
   
    def add_main_menu_actions(self, subparser):
        path_parser = subparser.add_parser('enevals', help='Search for ensemble evaluators')
        path_parser.add_argument('--filter', '-f', required=False, help='Search for ensemble evaluators relative to the base path, possibly specifying a filter', dest='filter')
                
        path_parser = subparser.add_parser('seneval', help='Select an ensemble evaluator')
        path_parser.add_argument('pos', help='Select an ensemble evaluator, specify position', type=int)
        path_parser.add_argument('--networks', '-n', required=False, help='Specifies the number of networks to use', dest='networks', type=int)
        
    def handle_command(self, cmd, command, base_path):
        if cmd == 'enevals':
            self._paths = EnsembleReporter.find_ensemble_runs(base_path)
            self._display_paths(base_path)
            return True
        elif cmd == 'seneval':
            if command.networks:
                if command.networks < 2:
                    print('At least two networks must be selected')
                    return
                self._networks = command.networks
            else:
                self._networks = None         
            self._select_actor(command, base_path)
            return True
        
    def _get_actor(self):
        return EnsembleReporter(self._path, num_networks=self._networks, overwrite=True).get_ensemble_evaluator(find_best_ensemble=True)
    
    def _get_menu_options(self):
        
        options = ['Show real absolute mean error',
                   'Show scaled absolute mean error',
                   'Show real relative mean error',
                   'Show scaled relative mean error',
                   'Plot target vs ensemble mean and best network real',
                   'Plot target vs ensemble mean and best network real with tail',
                   'Plot target vs ensemble mean and best network scaled',
                   'Plot target vs ensemble mean and best network scaled with tail',
                   'Plot target vs ensemble mean, min and max real',
                   'Plot target vs ensemble mean, min and max real with tail',
                   'Plot target vs ensemble mean, min and max scaled',
                   'Plot target vs ensemble mean, min and max scaled with tail',
                   'Plot target vs ensemble mean and interval real',
                   'Plot target vs ensemble mean and interval real with tail',
                   'Plot target vs ensemble mean and interval scaled',
                   'Plot target vs ensemble mean and interval scaled with tail',
                   'Plot real standard deviation',
                   'Plot scaled standard deviation',
                   'Plot variance errors',
                   'Plot real min max range',
                   'Plot scaled min max range']
        return super(EnsembleEvaluatorActionMenu, self)._get_menu_options() + options
                                    
    def _handle_action(self, action, feature_pos, params):
        
        if action == 8:
            print('{} absolute mean error: {:.2f} Best Network: {:.2f}'.format(self._actor.generate_feature_name(feature_pos, scaled=False), self._actor.absolute_mean_error(feature_pos), self._actor.best_network.absolute_mean_error(feature_pos) ))
        elif action == 9:
            print('{} absolute mean error: {:.2f} Best Network: {:.2f}'.format(self._actor.generate_feature_name(feature_pos, scaled=True), self._actor.absolute_mean_error(feature_pos, scaled=True), self._actor.best_network.absolute_mean_error(feature_pos, scaled=True) ))
        elif action == 10:
            print('{} relative mean error: {:.2f}% Best Network: {:.2f}%'.format(self._actor.generate_feature_name(feature_pos, scaled=False), self._actor.relative_mean_error(feature_pos), self._actor.best_network.relative_mean_error(feature_pos)))
        elif action == 11:
            print('{} relative mean error: {:.2f}% Best Network: {:.2f}%'.format(self._actor.generate_feature_name(feature_pos, scaled=True), self._actor.relative_mean_error(feature_pos, scaled=True), self._actor.best_network.relative_mean_error(feature_pos, scaled=True)))
        elif action == 12:
            self._actor.plot_target_vs_mean_best(feature_pos)
        elif action == 13:
            self._actor.plot_target_vs_mean_best(feature_pos, tail=True)
        elif action == 14:
            self._actor.plot_target_vs_mean_best(feature_pos, scaled=True)
        elif action == 15:
            self._actor.plot_target_vs_mean_best(feature_pos, scaled=True, tail=True)
        elif action == 16:
            self._actor.plot_target_vs_mean_min_max(feature_pos)
        elif action == 17:
            self._actor.plot_target_vs_mean_min_max(feature_pos, tail=True)
        elif action == 18:
            self._actor.plot_target_vs_mean_min_max(feature_pos, scaled=True)
        elif action == 19:
            self._actor.plot_target_vs_mean_min_max(feature_pos, scaled=True, tail=True)
        elif action == 20:
            self._actor.plot_target_vs_mean_interval(feature_pos)
        elif action == 21:
            self._actor.plot_target_vs_mean_interval(feature_pos, tail=True)
        elif action == 22:
            self._actor.plot_target_vs_mean_interval(feature_pos, scaled=True)
        elif action == 23:
            self._actor.plot_target_vs_mean_interval(feature_pos, scaled=True, tail=True)
        elif action == 24:
            self._actor.plot_std(feature_pos)
        elif action == 25:
            self._actor.plot_std(feature_pos, scaled=True)
        elif action == 26:
            self._actor.plot_variance_errors(feature_pos)
        elif action == 27:
            self._actor.plot_min_max_range(feature_pos)
        elif action == 28:
            self._actor.plot_min_max_range(feature_pos, scaled=True)
        else:
            raise ValueError('Unknown action')
            