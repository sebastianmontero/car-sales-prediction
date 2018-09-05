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
        self._path = None
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
        if command.pos >= 0 and command.pos < len(self._paths):
            self._path = self._paths[command.pos]
            self._actor = self._get_actor()
            print('Selected {}: {}'.format(self._title, self._path))
            self._enter_action_mode()
        else:
            print('Invalid position')
            self._display_paths(base_path)
            
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
            