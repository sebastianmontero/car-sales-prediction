'''
Created on Jul 2, 2018

@author: nishilab
'''
import os
import matplotlib.pyplot as plt
from evaluator import Evaluator
from storage_manager import StorageManager, StorageManagerType
from action_menu import ActionMenu

class EvaluatorActionMenu(ActionMenu):
    
    def __init__(self, config_sm):
        ActionMenu.__init__(self, 'Evaluator',StorageManager.get_storage_manager(StorageManagerType.EVALUATOR), config_sm)
   
    def add_main_menu_actions(self, subparser):
        path_parser = subparser.add_parser('evals', help='Search for evaluators')
        path_parser.add_argument('--filter', '-f', required=False, help='Search for evaluators relative to the base path, possibly specifying a filter', dest='filter')
        path_parser.add_argument('--show-windows', '-w', required=False, help='Indicates if it should show paths for window evaluators', dest='show_windows', action='store_true')
        
        path_parser = subparser.add_parser('seval', help='Select an evaluator')
        path_parser.add_argument('pos', help='Select an evaluator, specify position', type=int)
        
        path_parser = subparser.add_parser('vevals', help='Plot value for all evals')
        path_parser.add_argument('value', help='Value to plot ', type=str)
        
        
    def handle_command(self, cmd, command, base_path):
        if cmd == 'evals':
            exclude_filter = None if command.show_windows else 'w-\d{6}-\d{6}'
            self._paths = self._sm.get_pickles(base_path, command.filter, recursive=1, exclude_filter=exclude_filter)
            self._display_paths(base_path)
            return True
        elif cmd == 'seval':
            self._select_actor(command, base_path)
            return True
        elif cmd == 'vevals':
            self._evals_plot(command.value)
            return True
    
    def _evals_plot(self, value):
        
        if len(self.paths) == 0:
            print('No evals. evals command must be used before fevals command')
            return
        
        
        value_map = {
            'gs': {'fn':'global_step', 'name':'Global Step'},
            'rme': {'fn':'real_relative_mean_error', 'name':'Real Relative Mean Error'},
            'srme': {'fn':'scaled_relative_mean_error', 'name':'Scaled Relative Mean Error'},
            'ame': {'fn':'real_absolute_mean_error', 'name':'Real Absolute Mean Error'},
            'same': {'fn':'scaled_absolute_mean_error', 'name': 'Scaled Absolute Mean Error'}
        }
        
        obj = value_map[value]
        values = self._get_evals_value(obj['fn'])
        
        if values:
            plt.plot(values)
            plt.title(obj['name'])
            plt.show()
        
        
    def _get_evals_value(self, fn):
        vals = []
        
        try:
            for path in self.paths:
                evaluator = self._unpickle(path)
                attr = getattr(evaluator, fn)
                vals.append(attr() if callable(attr) else attr)
        except AttributeError:
            print('Evaluator values do not have the requested value')
            return None
        return vals
            
    def _unpickle(self, path):
        return self._sm.unpickle(path)
    
    def _get_actor(self):
        return self._unpickle(self._path)
    
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
        print('[11] Show related configuration')
                                    
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
            print('Real sales absolute mean error: {:.2f}'.format(self._actor.real_absolute_mean_error()))
        elif action == 8:
            print('Scaled sales absolute mean error: {:.5f}'.format(self._actor.scaled_absolute_mean_error()))
        elif action == 9:
            print('Real sales relative mean error: {:.2f}%'.format(self._actor.real_relative_mean_error()))
        elif action == 10:
            print('Scaled sales relative mean error: {:.2f}%'.format(self._actor.scaled_relative_mean_error()))
        elif action == 11:
            path,_ = os.path.split(self._path)
            config = self._config_sm.unpickle(path)
            print('Configuration: ')
            print()
            self._pprint.pprint(config)
        else:
            raise ValueError('Unknown action')
            