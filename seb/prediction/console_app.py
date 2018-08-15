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
from feature_selector_reporter import FeatureSelectorReporter
from ensemble_reporter import EnsembleReporter

class ConsoleApp():
    
    def __init__(self):
        
        #self._base_path = os.path.dirname(os.path.realpath(__file__))
        self._base_path = '/home/nishilab/Documents/python/model-storage/'
        self._parser = self._create_parser()
        self._evaluators = []
        self._fss = []
        self._ensemble_evaluators = []
        self._evaluator = None
        self._ensemble_evaluator = None
        self._fs_reporter = None
        self._evaluator_path = None
        self._fs_path = None
        self._ensemble_evaluator_path = None
        self._evaluator_sm = StorageManager.get_storage_manager(StorageManagerType.EVALUATOR)
        self._config_sm = StorageManager.get_storage_manager(StorageManagerType.CONFIG)
        self._ensemble_evaluator_sm = StorageManager.get_storage_manager(StorageManagerType.ENSEMBLE_EVALUATOR)
        self._pprint = pprint.PrettyPrinter()
        
    def _create_parser(self):
        parser = argparse.ArgumentParser(description="Evaluate modules")
        subparser = parser.add_subparsers(help='sub-command help', dest='cmd')
        subparser.add_parser('exit', help='Exit application')
        
        path_parser = subparser.add_parser('path', help='Change base path, if a path is not specified the current path is shown')
        path_parser.add_argument('--path', '-p', required=False, help='Sets the base path to search from', dest='path')
        
        path_parser = subparser.add_parser('evals', help='Search for evaluators')
        path_parser.add_argument('--filter', '-f', required=False, help='Search for evaluators relative to the base path, possibly specifying a filter', dest='filter')
        path_parser.add_argument('--show-windows', '-w', required=False, help='Indicates if it should show paths for window evaluators', dest='show_windows', action='store_true')
        
        path_parser = subparser.add_parser('fs', help='Search for feature selection runs')
        
        path_parser = subparser.add_parser('enevals', help='Search for ensemble evaluators')
        path_parser.add_argument('--filter', '-f', required=False, help='Search for ensemble evaluators relative to the base path, possibly specifying a filter', dest='filter')
        
        path_parser = subparser.add_parser('seval', help='Select an evaluator')
        path_parser.add_argument('pos', help='Select an evaluator, specify position', type=int)
        
        path_parser = subparser.add_parser('sfs', help='Select a feature selector run')
        path_parser.add_argument('pos', help='Select a feature selector run, specify position', type=int)
        
        path_parser = subparser.add_parser('seneval', help='Select an ensemble evaluator')
        path_parser.add_argument('pos', help='Select an ensemble evaluator, specify position', type=int)
        
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
        if cmd == 'evals':
            exclude_filter = None if command.show_windows else 'w-\d{6}-\d{6}'
            self._evaluators = self._evaluator_sm.get_pickles(self._base_path, command.filter, recursive=True, exclude_filter=exclude_filter)
            self._display_evaluators()
        if cmd == 'enevals':
            self._ensemble_evaluators = EnsembleReporter.find_ensemble_runs(self._base_path)
            self._display_ensemble_evaluators()
        if cmd == 'fs':
            self._fss = FeatureSelectorReporter.find_feature_selector_runs(self._base_path)
            self._display_feature_selectors()
        if cmd == 'seval':
            if command.pos >= 0 and command.pos < len(self._evaluators):
                self._evaluator_path = self._evaluators[command.pos]
                self._evaluator = self._evaluator_sm.unpickle(self._evaluator_path)
                print('Selected Evaluator:', self._evaluator_path)
                self._evaluator_mode()
            else:
                print('Invalid evaluator position')
                self._display_evaluators()
        if cmd == 'seneval':
            if command.pos >= 0 and command.pos < len(self._ensemble_evaluators):
                self._ensemble_evaluator_path = self._ensemble_evaluators[command.pos]
                self._ensemble_evaluator = EnsembleReporter(self._ensemble_evaluator_path).get_ensemble_evaluator()
                print('Selected Ensemble Evaluator:', self._ensemble_evaluator_path)
                self._ensemble_evaluator_mode()
            else:
                print('Invalid evaluator position')
                self._display_ensemble_evaluators()
        if cmd == 'sfs':
            if command.pos >= 0 and command.pos < len(self._fss):
                self._fs_path = self._fss[command.pos]
                self._fs_reporter = FeatureSelectorReporter(run_path=self._fs_path)
                print('Selected Feature Selector:', self._fs_path)
                self._feature_selector_mode()
            else:
                print('Invalid evaluator position')
                self._display_evaluators()
    
    def _display_evaluators(self):
        self._display_paths(self._evaluators, 'Evaluators')
        
    def _display_ensemble_evaluators(self):
        self._display_paths(self._ensemble_evaluators, 'Ensemble Evaluators')
    
    def _display_feature_selectors(self):
        self._display_paths(self._fss, 'Feature Selectors')
            
    def _display_paths(self, paths, title):
        base_path_pos = len(self._base_path)
        print()
        print(title + ':')
        for pos, path in enumerate(paths):
            print('[{}] {}'.format(pos, path[base_path_pos:]))    
        
    def _print_evaluator_menu(self):
        print()
        print('Evaluator mode options:')
        print()
        print('[0] Exit evaluator mode')
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
        print()
        
    def _print_ensemble_evaluator_menu(self):
        print()
        print('Evaluator mode options:')
        print()
        print('[0] Exit evaluator mode')
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
        print()
    
    def _print_feature_selector_menu(self):
        print()
        print('Feature selector mode options:')
        print()
        print('[0] Exit feature selector mode')
        print('[1] Show best configuration')
        print('[2] Show best configuration per feature')
        print('[3] Show configurations per feature [Specify number of features wanted]')
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
                
    def _ensemble_evaluator_mode(self):
        while True:
            self._print_ensemble_evaluator_menu()
            try:
                action = int(input('Select an option \n>>> '))
                print()
                if action == 0:
                    break;
                
                self._perform_ensemble_evaluator_action(action)
            except ValueError:
                print('Invalid option')
    
    def _feature_selector_mode(self):
        while True:
            self._print_feature_selector_menu()
            try:
                action = input('Select an option \n>>> ')
                split_action = action.split()
                action = int(split_action[0])

                print()
                if action == 0:
                    break;
                
                self._perform_feature_selector_action(action, split_action)
            except (ValueError, IndexError):
                print('Invalid option')
                
    def _perform_evaluator_action(self, action):
        
        if action == 1:
            self._evaluator.plot_real_target_vs_predicted()
        if action == 2:
            self._evaluator.plot_real_target_vs_predicted(tail=True)
        elif action == 3:
            self._evaluator.plot_scaled_target_vs_predicted()
        elif action == 4:
            self._evaluator.plot_scaled_target_vs_predicted(tail=True)
        elif action == 5:
            self._evaluator.plot_real_errors()
        elif action == 6:
            self._evaluator.plot_scaled_errors()
        elif action == 7:
            print('Real sales absolute mean error: {:.2f}'.format(self._evaluator.real_absolute_mean_error()))
        elif action == 8:
            print('Scaled sales absolute mean error: {:.5f}'.format(self._evaluator.scaled_absolute_mean_error()))
        elif action == 9:
            print('Real sales relative mean error: {:.2f}%'.format(self._evaluator.real_relative_mean_error()))
        elif action == 10:
            print('Scaled sales relative mean error: {:.2f}%'.format(self._evaluator.scaled_relative_mean_error()))
        elif action == 11:
            path,_ = os.path.split(self._evaluator_path)
            config = self._config_sm.unpickle(path)
            print('Configuration: ')
            print()
            self._pprint.pprint(config)
        else:
            raise ValueError('Unknown action')
        
    def _perform_ensemble_evaluator_action(self, action):
        
        if action == 1:
            self._ensemble_evaluator.plot_real_target_vs_predicted()
        if action == 2:
            self._ensemble_evaluator.plot_real_target_vs_predicted(tail=True)
        elif action == 3:
            self._ensemble_evaluator.plot_scaled_target_vs_predicted()
        elif action == 4:
            self._ensemble_evaluator.plot_scaled_target_vs_predicted(tail=True)
        elif action == 5:
            self._ensemble_evaluator.plot_real_errors()
        elif action == 6:
            self._ensemble_evaluator.plot_scaled_errors()
        elif action == 7:
            print('Real sales absolute mean error: {:.2f} Best Network: {:.2f}'.format(self._ensemble_evaluator.real_absolute_mean_error(), self._ensemble_evaluator.best_network.real_absolute_mean_error()))
        elif action == 8:
            print('Scaled sales absolute mean error: {:.5f} Best Network: {:.5f}'.format(self._ensemble_evaluator.scaled_absolute_mean_error(), self._ensemble_evaluator.best_network.scaled_absolute_mean_error()))
        elif action == 9:
            print('Real sales relative mean error: {:.2f}% Best Network: {:.2f}'.format(self._ensemble_evaluator.real_relative_mean_error()), self._ensemble_evaluator.best_network.real_relative_mean_error())
        elif action == 10:
            print('Scaled sales relative mean error: {:.2f}% Best Network: {:.2f}%'.format(self._ensemble_evaluator.scaled_relative_mean_error(), self._ensemble_evaluator.best_network.scaled_relative_mean_error()))
        elif action == 11:
            self._ensemble_evaluator.plot_real_target_vs_mean_best()
        elif action == 12:
            self._ensemble_evaluator.plot_real_target_vs_mean_best(tail=True)
        elif action == 13:
            self._ensemble_evaluator.plot_scaled_target_vs_mean_best()
        elif action == 14:
            self._ensemble_evaluator.plot_scaled_target_vs_mean_best(tail=True)
        elif action == 15:
            self._ensemble_evaluator.plot_real_target_vs_mean_min_max()
        elif action == 16:
            self._ensemble_evaluator.plot_real_target_vs_mean_min_max(tail=True)
        elif action == 17:
            self._ensemble_evaluator.plot_scaled_target_vs_mean_min_max()
        elif action == 18:
            self._ensemble_evaluator.plot_scaled_target_vs_mean_min_max(tail=True)
        elif action == 19:
            self._ensemble_evaluator.plot_real_target_vs_mean_interval()
        elif action == 20:
            self._ensemble_evaluator.plot_real_target_vs_mean_interval(tail=True)
        elif action == 21:
            self._ensemble_evaluator.plot_scaled_target_vs_mean_interval()
        elif action == 22:
            self._ensemble_evaluator.plot_scaled_target_vs_mean_interval(tail=True)
        else:
            raise ValueError('Unknown action')
                            
    def _perform_feature_selector_action(self, action, params):
        
        if action == 1:
            self._fs_reporter.print_best_config()
        elif action == 2:
            self._fs_reporter.print_best_configs()
        elif action == 3:
            self._fs_reporter.print_experiment_configs(int(params[1]))
        else:
            raise ValueError('Unknown action')
            
        
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