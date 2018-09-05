'''
Created on Jul 2, 2018

@author: nishilab
'''
import pprint
import traceback
from evaluator import Evaluator
from storage_manager import StorageManager, StorageManagerType
from feature_selector_reporter import FeatureSelectorReporter
from ensemble_reporter import EnsembleReporter

class ActionMenu():
    
    def __init__(self, title, sm, config_sm):
        
        self._title = title
        self._paths = []
        self._actor = None
        self._sel_paths = None
        self._sm = sm
        self._config_sm = config_sm
        self._pprint = pprint.PrettyPrinter()
        
    @property
    def paths(self):
        return self._paths
   
    def add_main_menu_actions(self, subparser):
        raise NotImplementedError("Subclasses must implement this method")
        
    def handle_command(self, cmd, command, base_path):
        raise NotImplementedError("Subclasses must implement this method")
    
    def _display_paths(self, base_path):
        base_path_pos = len(base_path)
        print()
        print(self._title + 's:')
        for pos, path in enumerate(self._paths):
            print('[{}] {}'.format(pos, path[base_path_pos:]))    
        
    def _print_menu(self):
        print()
        print('{} mode options:'.format(self._title))
        print()
        print('[0] Exit {} mode'.format(self._title))
        
        options = self._get_menu_options()
        for i, option in enumerate(options):
            print('[{}] {}'.format(i + 1, option))
        print()
    
    def _get_menu_options(self):
        raise NotImplementedError("Subclasses must implement this method")
    
        
    def _select_actor(self, command, base_path):
        sel_paths = []
        
        for pos in command.pos:
            
            if pos >= 0 and pos < len(self._paths):
                sel_paths.append(self._paths[pos])
            else:
                print('Invalid position:', pos)
                self._display_paths(base_path)
                return
        
        self._sel_paths = sel_paths    
        self._actor = self._get_actor()
        print('Selected {}:'.format(self._title))
        
        for sel_path in sel_paths:
            print(sel_path)
            
        self._enter_action_mode()
        
            
    def _get_actor(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def _enter_action_mode(self):
        while True:
            self._print_menu()
            try:
                action = input('Select an option \n>>> ')
                split_action = action.split()
                action = int(split_action[0])

                print()
                if action == 0:
                    break;
                
                self._perform_action(action, split_action)
            except (ValueError, IndexError) as e:
                traceback.print_exc()
                print('Invalid option')
                                    
    def _perform_action(self, action, params):
        raise NotImplementedError("Subclasses must implement this method")
            