'''
Created on Jul 2, 2018

@author: nishilab
'''
import os
from ensemble_evaluator import EnsembleReporter
from storage_manager import StorageManager, StorageManagerType
from action_menu import ActionMenu

class EnsembleEvaluatorActionMenu(ActionMenu):
    
    def __init__(self, config_sm):
        ActionMenu.__init__(self, 'Ensemble Evaluator',StorageManager.get_storage_manager(StorageManagerType.ENSEMBLE_EVALUATOR), config_sm)
   
    def add_main_menu_actions(self, subparser):
        path_parser = subparser.add_parser('enevals', help='Search for ensemble evaluators')
        path_parser.add_argument('--filter', '-f', required=False, help='Search for ensemble evaluators relative to the base path, possibly specifying a filter', dest='filter')
                
        path_parser = subparser.add_parser('seneval', help='Select an ensemble evaluator')
        path_parser.add_argument('pos', help='Select an ensemble evaluator, specify position', type=int)
        
        
    def handle_command(self, cmd, command, base_path):
        if cmd == 'enevals':
            self._paths = EnsembleReporter.find_ensemble_runs(base_path)
            self._display_paths(base_path)
            return True
        elif cmd == 'seneval':
            self._select_actor(command, base_path)
            return True
        
    def _get_actor(self):
        return EnsembleReporter(self._path).get_ensemble_evaluator()
    
    def _print_menu_options(self):
        print('[1] Plot target vs predicted real sales')
        print('[2] Plot target vs predicted real sales with tail')
        print('[3] Plot target vs predicted scaled sales')
        print('[4] Plot target vs predicted scaled sales with tail')
        print('[5] Plot real sales errors')
        print('[6] Plot scaled sales errors')
        print('[7] Show real sales absolute mean error')
        print('[8] Show scaled sales absolute mean error')
        print('[9] Show real sales relative mean error')
        print('[10] Show scaled sales relative mean error')
        print('[11] Plot target vs ensemble mean and best network real sales')
        print('[12] Plot target vs ensemble mean and best network real sales with tail')
        print('[13] Plot target vs ensemble mean and best network scaled sales')
        print('[14] Plot target vs ensemble mean and best network scaled sales with tail')
        print('[15] Plot target vs ensemble mean, min and max real sales')
        print('[16] Plot target vs ensemble mean, min and max real sales with tail')
        print('[17] Plot target vs ensemble mean, min and max scaled sales')
        print('[18] Plot target vs ensemble mean, min and max scaled sales with tail')
        print('[19] Plot target vs ensemble mean and interval real sales')
        print('[20] Plot target vs ensemble mean and interval real sales with tail')
        print('[21] Plot target vs ensemble mean and interval scaled sales')
        print('[22] Plot target vs ensemble mean and interval scaled sales with tail')
                                    
    def _perform_action(self, action, params):
        if action == 1:
            self._actor.plot_real_target_vs_predicted()
        if action == 2:
            self._actor.plot_real_target_vs_predicted(tail=True)
        elif action == 3:
            self._actor.plot_scaled_target_vs_predicted()
        elif action == 4:
            self._actor.plot_scaled_target_vs_predicted(tail=True)
        elif action == 5:
            self._actor.plot_real_errors()
        elif action == 6:
            self._actor.plot_scaled_errors()
        elif action == 7:
            print('Real sales absolute mean error: {:.2f} Best Network: {:.2f}'.format(self._actor.real_absolute_mean_error(), self._actor.best_network.real_absolute_mean_error()))
        elif action == 8:
            print('Scaled sales absolute mean error: {:.5f} Best Network: {:.5f}'.format(self._actor.scaled_absolute_mean_error(), self._actor.best_network.scaled_absolute_mean_error()))
        elif action == 9:
            print('Real sales relative mean error: {:.2f}% Best Network: {:.2f}%'.format(self._actor.real_relative_mean_error(), self._actor.best_network.real_relative_mean_error()))
        elif action == 10:
            print('Scaled sales relative mean error: {:.2f}% Best Network: {:.2f}%'.format(self._actor.scaled_relative_mean_error(), self._actor.best_network.scaled_relative_mean_error()))
        elif action == 11:
            self._actor.plot_real_target_vs_mean_best()
        elif action == 12:
            self._actor.plot_real_target_vs_mean_best(tail=True)
        elif action == 13:
            self._actor.plot_scaled_target_vs_mean_best()
        elif action == 14:
            self._actor.plot_scaled_target_vs_mean_best(tail=True)
        elif action == 15:
            self._actor.plot_real_target_vs_mean_min_max()
        elif action == 16:
            self._actor.plot_real_target_vs_mean_min_max(tail=True)
        elif action == 17:
            self._actor.plot_scaled_target_vs_mean_min_max()
        elif action == 18:
            self._actor.plot_scaled_target_vs_mean_min_max(tail=True)
        elif action == 19:
            self._actor.plot_real_target_vs_mean_interval()
        elif action == 20:
            self._actor.plot_real_target_vs_mean_interval(tail=True)
        elif action == 21:
            self._actor.plot_scaled_target_vs_mean_interval()
        elif action == 22:
            self._actor.plot_scaled_target_vs_mean_interval(tail=True)
        else:
            raise ValueError('Unknown action')
            