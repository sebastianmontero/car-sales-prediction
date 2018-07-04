'''
Created on Jul 2, 2018

@author: nishilab
'''

import os
import sys
import argparse
import pprint
from evaluator import Evaluator
from storage_manager import StorageManager, StorageManagerType

class ConsoleApp():
    
    def __init__(self):
        
        #self._base_path = os.path.dirname(os.path.realpath(__file__))
        self._base_path = '/home/nishilab/Documents/python/model-storage/best_feature'
        self._parser = self._create_parser()
        self._evaluators = []
        self._evaluator = None
        self._evaluator_path = None
        self._evaluator_sm = StorageManager.get_storage_manager(StorageManagerType.EVALUATOR)
        self._config_sm = StorageManager.get_storage_manager(StorageManagerType.CONFIG)
        self._pprint = pprint.PrettyPrinter()
        
    def _create_parser(self):
        parser = argparse.ArgumentParser(description="Evaluate modules")
        subparser = parser.add_subparsers(help='sub-command help', dest='cmd')
        subparser.add_parser('exit', help='Exit application')
        
        path_parser = subparser.add_parser('path', help='Change base path, if a path is not specified the current path is shown')
        path_parser = path_parser.add_argument('--path', '-p', required=False, help='Sets the base path to search from', dest='path')
        
        path_parser = subparser.add_parser('search', help='Search for evaluators')
        path_parser = path_parser.add_argument('--filter', '-f', required=False, help='Search for evaluators relative to the base path, possibly specifying a filter', dest='filter')
        
        path_parser = subparser.add_parser('select', help='Select an evaluator')
        path_parser = path_parser.add_argument('pos', help='Select an evaluator, specify position', type=int)
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
        if cmd == 'path':
            if hasattr(command, 'path') and command.path:
                if os.path.isdir(command.path):
                    self._base_path = command.path
                else:
                    print('Invalid path: ', command.path)
            else:    
                print('Base path: ', self._base_path)
        if cmd == 'search':
            self._search(command)
        if cmd == 'select':
            if command.pos >= 0 and command.pos < len(self._evaluators):
                self._evaluator_path = self._evaluators[command.pos]
                self._evaluator = self._evaluator_sm.unpickle(self._evaluator_path)
                print('Selected Evaluator:', self._evaluator_path)
                self._evaluator_mode()
            else:
                print('Invalid evaluator position')
                self._display_evaluators()
    
    def _display_evaluators(self):
        base_path_pos = len(self._base_path)
        print()
        print('Evaluators:')
        for pos, evaluator in enumerate(self._evaluators):
            print('[{}] {}'.format(pos, evaluator[base_path_pos:]))
    
    def _search(self, command): 
        
        self._evaluators = self._evaluator_sm.get_pickles(self._base_path, command.filter, recursive=True)
        self._display_evaluators() 
        
        
    def _print_evaluator_menu(self):
        print()
        print('Evaluator mode options:')
        print()
        print('[0] Exit evaluator mode')
        print('[1] Plot target vs predicted real sales')
        print('[2] Plot target vs predicted scaled sales')
        print('[3] Plot real sales errors')
        print('[4] Plot scaled sales errors')
        print('[5] Show real sales absolute mean error')
        print('[6] Show scaled sales absolute mean error')
        print('[7] Show real sales relative mean error')
        print('[8] Show scaled sales relative mean error')
        print('[9] Show related configuration')
        print()
        
        
    def _evaluator_mode(self):
        while True:
            self._print_evaluator_menu()
            try:
                action = int(input('Select an option \n>>> '))
                print()
                if action == 0:
                    break;
                
                self._perform_evaluator_action(action)
            except ValueError:
                print('Invalid option')
                
    def _perform_evaluator_action(self, action):
        
        if action == 1:
            self._evaluator.plot_real_target_vs_predicted()
        elif action == 2:
            self._evaluator.plot_scaled_target_vs_predicted()
        elif action == 3:
            self._evaluator.plot_real_errors()
        elif action == 4:
            self._evaluator.plot_scaled_errors()
        elif action == 5:
            print('Real sales absolute mean error: {:.2f}'.format(self._evaluator.real_absolute_mean_error()))
        elif action == 6:
            print('Scaled sales absolute mean error: {:.5f}'.format(self._evaluator.scaled_absolute_mean_error()))
        elif action == 7:
            print('Real sales relative mean error: {:.2f}'.format(self._evaluator.real_relative_mean_error()))
        elif action == 8:
            print('Scaled sales relative mean error: {:.2f}'.format(self._evaluator.scaled_relative_mean_error()))
        elif action == 9:
            path,_ = os.path.split(self._evaluator_path)
            config = self._config_sm.unpickle(path)
            print('Configuration: ')
            print()
            self._pprint.pprint(config)
            
        
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