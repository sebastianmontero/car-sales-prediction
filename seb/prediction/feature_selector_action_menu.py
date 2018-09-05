'''
Created on Jul 2, 2018

@author: nishilab
'''
from storage_manager import StorageManager, StorageManagerType
from feature_selector_reporter import FeatureSelectorReporter
from action_menu import ActionMenu

class FeatureSelectorActionMenu(ActionMenu):
    
    def __init__(self, config_sm):
        ActionMenu.__init__(self, 'Feature Selector',config_sm, config_sm)
   
    def add_main_menu_actions(self, subparser):
        subparser.add_parser('fs', help='Search for feature selection runs')
        
        path_parser = subparser.add_parser('sfs', help='Select a feature selector run')
        path_parser.add_argument('pos', help='Select a feature selector run, specify position', type=int, nargs='+')
        
    def handle_command(self, cmd, command, base_path):
        if cmd == 'fs':
            self._paths = FeatureSelectorReporter.find_feature_selector_runs(base_path)
            self._display_paths(base_path)
            return True
        elif cmd == 'sfs':
            self._select_actor(command, base_path)
            return True
        
    def _get_actor(self):
        return FeatureSelectorReporter(run_path=self._sel_paths[0])
        
    def _print_menu_options(self):
        print('[1] Show best configuration')
        print('[2] Show best configuration per feature')
        print('[3] Show configurations per feature [Specify number of features wanted]')
                                        
    def _perform_action(self, action, command):
        if action == 1:
            self._actor.print_best_config()
        elif action == 2:
            self._actor.print_best_configs()
        elif action == 3:
            self._actor.print_experiment_configs(command.num_features)
        else:
            raise ValueError('Unknown action')
            