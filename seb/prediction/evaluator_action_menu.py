'''
Created on Jul 2, 2018

@author: nishilab
'''
import os
import matplotlib.pyplot as plt
from evaluator_presenter import EvaluatorPresenter
from storage_manager import StorageManager, StorageManagerType
from base_evaluator_action_menu import BaseEvaluatorActionMenu

class EvaluatorActionMenu(BaseEvaluatorActionMenu):
    
    def __init__(self, config_sm):
        BaseEvaluatorActionMenu.__init__(self, 'Evaluator',StorageManager.get_storage_manager(StorageManagerType.EVALUATOR), config_sm)
   
    def add_main_menu_actions(self, subparser):
        path_parser = subparser.add_parser('evals', help='Search for evaluators')
        path_parser.add_argument('--filter', '-f', required=False, help='Search for evaluators relative to the base path, possibly specifying a filter', dest='filter')
        path_parser.add_argument('--show-windows', '-w', required=False, help='Indicates if it should show paths for window evaluators', dest='show_windows', action='store_true')
        
        path_parser = subparser.add_parser('seval', help='Select an evaluator')
        path_parser.add_argument('pos', help='Select an evaluator, specify position', type=int, nargs='+')
        
        path_parser = subparser.add_parser('vevals', help='Plot value for all evals')
        path_parser.add_argument('value', help='Value to plot ', type=str)
        
        
    def handle_command(self, cmd, command, base_path):
        if cmd == 'evals':
            exclude_filter = None if command.show_windows else 'w-\d{6}-\d{6}'
            self._paths = self._sm.get_pickles(base_path, command.filter, recursive='car_sales_prediction_trainable_', exclude_filter=exclude_filter)
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
        evals = []
        
        for path in self._sel_paths:
            evals.append({
                'name':os.path.basename(path),
                'obj': self._unpickle(path)
            })
        return EvaluatorPresenter(evals)
    
    def _get_menu_options(self):
        options = ['Show related configuration']
        return super(EvaluatorActionMenu, self)._get_menu_options() + options
                                    
    def _handle_action(self, action, *_):
        if action == 9:
            path,_ = os.path.split(self._path)
            config = self._config_sm.unpickle(path)
            print('Configuration: ')
            print()
            self._pprint.pprint(config)
        else:
            raise ValueError('Unknown action')
            