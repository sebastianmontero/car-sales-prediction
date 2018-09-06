'''
Created on Jul 2, 2018

@author: nishilab
'''
import pprint
import traceback
import argparse
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
        self._parser = self._create_parser()
        self._pprint = pprint.PrettyPrinter()
        
    @property
    def paths(self):
        return self._paths
    
    def _create_parser(self):
        parser = argparse.ArgumentParser(description="Evaluate " + self._title)
        subparser = parser.add_subparsers(help='sub-command help', dest='cmd')
        subparser.add_parser('exit', help='Exit application')
        
        path_parser = subparser.add_parser('sel', help='Select action to perform')
        path_parser.add_argument('pos', help='Number corresponding to the action to perform', type=int)
        path_parser.add_argument('--feature', '-f', required=False, help='Sets the feature to use', dest='feature', type=int, default=0)
        path_parser.add_argument('--evals', '-e', required=False, help='Select the evals to use', dest='evals', type=int, default=[], nargs='+')
        path_parser.add_argument('--num', '-n', required=False, help='Number of features', dest='num_features', type=int, default=1)
        path_parser.add_argument('--scaled', '-s', required=False, help='Use scaled values', dest='scaled', action='store_true')
        path_parser.add_argument('--tail', '-t', required=False, help='Show real values tail', dest='tail', action='store_true')        
       
        return parser;
   
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
                command = self._parser.parse_args(action.split())
                print()
                cmd = command.cmd
                if cmd == 'exit':
                    break;
                elif cmd == 'sel':
                    self._perform_action(command.pos, command)
            except:
                #traceback.print_exc()
                print('Invalid option')
                                    
    def _perform_action(self, action, feature, evals):
        raise NotImplementedError("Subclasses must implement this method")
            