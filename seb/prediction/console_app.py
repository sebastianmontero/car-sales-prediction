'''
Created on Jul 2, 2018

@author: nishilab
'''

import os
import sys
import argparse
import pprint
from storage_manager import StorageManager, StorageManagerType
from evaluator_action_menu import EvaluatorActionMenu
from feature_selector_action_menu import FeatureSelectorActionMenu
from ensemble_evaluator_action_menu import EnsembleEvaluatorActionMenu

class ConsoleApp():
    
    def __init__(self):
        
        #self._base_path = os.path.dirname(os.path.realpath(__file__))
        self._base_path = '/home/nishilab/Documents/python/model-storage/'
        self._config_sm = StorageManager.get_storage_manager(StorageManagerType.CONFIG)
        self._evaluator_am = EvaluatorActionMenu(self._config_sm)
        self._ensemble_evaluator_am = EnsembleEvaluatorActionMenu(self._config_sm)
        self._feature_selector_am = FeatureSelectorActionMenu(self._config_sm)
        self._parser = self._create_parser()
        self._pprint = pprint.PrettyPrinter()
        
    def _create_parser(self):
        parser = argparse.ArgumentParser(description="Evaluate modules")
        subparser = parser.add_subparsers(help='sub-command help', dest='cmd')
        subparser.add_parser('exit', help='Exit application')
        
        path_parser = subparser.add_parser('path', help='Change base path, if a path is not specified the current path is shown')
        path_parser.add_argument('--path', '-p', required=False, help='Sets the base path to search from', dest='path')        
        
        self._evaluator_am.add_main_menu_actions(subparser)
        self._feature_selector_am.add_main_menu_actions(subparser)
        self._ensemble_evaluator_am.add_main_menu_actions(subparser)
        
        return parser;
        
    def _parse_action(self, action):
        try:
            return self._parser.parse_args(action.split())
        except SystemExit:
            print()
            
    def _perform_action(self, command):
        
        if not hasattr(command, 'cmd'):
            return
        
        cmd = command.cmd
        if cmd == 'exit':
            print('Good bye!')
            print()
            sys.exit()
        elif cmd == 'path':
            if hasattr(command, 'path') and command.path:
                if os.path.isdir(command.path):
                    self._base_path = command.path
                else:
                    print('Invalid path: ', command.path)
            else:    
                print('Base path: ', self._base_path)
        self._evaluator_am.handle_command(cmd, command, self._base_path)
        self._feature_selector_am.handle_command(cmd, command, self._base_path)
        self._ensemble_evaluator_am.handle_command(cmd, command, self._base_path)
        
    def run(self):
        action = ''
        
        while True:
            print()
            action = input('What would you like to do? \n>>> ')
            print()
            self._perform_action(self._parse_action(action))



if __name__ == '__main__':
    console_app = ConsoleApp()
    console_app.run()