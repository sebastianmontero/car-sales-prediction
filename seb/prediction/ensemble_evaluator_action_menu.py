'''
Created on Jul 2, 2018

@author: nishilab
'''
import os
from ensemble_reporter import EnsembleReporter
from storage_manager import StorageManager, StorageManagerType
from base_evaluator_action_menu import BaseEvaluatorActionMenu
from ensemble_evaluator_presenter import EnsembleEvaluatorPresenter

class EnsembleEvaluatorActionMenu(BaseEvaluatorActionMenu):
    
    def __init__(self, config_sm):
        BaseEvaluatorActionMenu.__init__(self, 'Ensemble Evaluator',StorageManager.get_storage_manager(StorageManagerType.ENSEMBLE_EVALUATOR), config_sm)
   
    def add_main_menu_actions(self, subparser):
        path_parser = subparser.add_parser('enevals', help='Search for ensemble evaluators')
        path_parser.add_argument('--filter', '-f', required=False, help='Search for ensemble evaluators relative to the base path, possibly specifying a filter', dest='filter')
                
        path_parser = subparser.add_parser('seneval', help='Select an ensemble evaluator')
        path_parser.add_argument('pos', help='Select an ensemble evaluator, specify position', type=int, nargs='+')
        path_parser.add_argument('--networks', '-n', required=False, help='Specifies the number of networks to use', dest='networks', type=int)
        path_parser.add_argument('--operator', '-o', required=False, help='Specifies the operator to use to ensemble networks', dest='operator', type=str, default='mean')
        
    def handle_command(self, cmd, command, base_path):
        if cmd == 'enevals':
            self._paths = EnsembleReporter.find_ensemble_runs(base_path)
            self._display_paths(base_path)
            return True
        elif cmd == 'seneval':
            if command.networks:
                if command.networks < 2:
                    print('At least two networks must be selected')
                    return
                self._networks = command.networks
            else:
                self._networks = None
                
            self._operator = command.operator         
            self._select_actor(command, base_path)
            return True
        
    def _get_actor(self):
        evals = []
        
        for path in self._sel_paths:
            evals.append({
                'name':os.path.basename(path),
                'obj': EnsembleReporter(path, num_networks=self._networks, overwrite=True).get_ensemble_evaluator(self._operator, find_best_ensemble=(self._networks is None))
            })
        return EnsembleEvaluatorPresenter(evals)
    
    def _get_menu_options(self):
        
        options = ['Show real absolute mean error',
                   'Show scaled absolute mean error',
                   'Show real relative mean error',
                   'Show scaled relative mean error',
                   'Plot target vs ensemble and best network real',
                   'Plot target vs ensemble and best network real with tail',
                   'Plot target vs ensemble and best network scaled',
                   'Plot target vs ensemble and best network scaled with tail',
                   'Plot target vs ensemble, min and max real',
                   'Plot target vs ensemble, min and max real with tail',
                   'Plot target vs ensemble, min and max scaled',
                   'Plot target vs ensemble, min and max scaled with tail',
                   'Plot target vs ensemble mean and interval real',
                   'Plot target vs ensemble mean and interval real with tail',
                   'Plot target vs ensemble mean and interval scaled',
                   'Plot target vs ensemble mean and interval scaled with tail',
                   'Plot real standard deviation',
                   'Plot scaled standard deviation',
                   'Plot variance errors',
                   'Plot real min max range',
                   'Plot scaled min max range']
        return super(EnsembleEvaluatorActionMenu, self)._get_menu_options() + options
                                    
    def _handle_action(self, action, feature_pos, evals):
        
        if action == 9:
            print(self._actor.absolute_mean_error_str(feature_pos))
        elif action == 10:
            print(self._actor.absolute_mean_error_str(feature_pos, scaled=True))
        elif action == 11:
            print(self._actor.relative_mean_error_str(feature_pos))
        elif action == 12:
            print(self._actor.relative_mean_error_str(feature_pos, scaled=True))
        elif action == 13:
            self._actor.plot_target_vs_ensemble_best(feature_pos)
        elif action == 14:
            self._actor.plot_target_vs_ensemble_best(feature_pos, tail=True)
        elif action == 15:
            self._actor.plot_target_vs_ensemble_best(feature_pos, scaled=True)
        elif action == 16:
            self._actor.plot_target_vs_ensemble_best(feature_pos, scaled=True, tail=True)
        elif action == 17:
            self._actor.plot_target_vs_ensemble_min_max(feature_pos)
        elif action == 18:
            self._actor.plot_target_vs_ensemble_min_max(feature_pos, tail=True)
        elif action == 19:
            self._actor.plot_target_vs_ensemble_min_max(feature_pos, scaled=True)
        elif action == 20:
            self._actor.plot_target_vs_ensemble_min_max(feature_pos, scaled=True, tail=True)
        elif action == 21:
            self._actor.plot_target_vs_mean_interval(feature_pos)
        elif action == 22:
            self._actor.plot_target_vs_mean_interval(feature_pos, tail=True)
        elif action == 23:
            self._actor.plot_target_vs_mean_interval(feature_pos, scaled=True)
        elif action == 24:
            self._actor.plot_target_vs_mean_interval(feature_pos, scaled=True, tail=True)
        elif action == 25:
            self._actor.plot_std(feature_pos)
        elif action == 26:
            self._actor.plot_std(feature_pos, scaled=True)
        elif action == 27:
            self._actor.plot_variance_errors(feature_pos)
        elif action == 28:
            self._actor.plot_min_max_range(feature_pos)
        elif action == 29:
            self._actor.plot_min_max_range(feature_pos, scaled=True)
        else:
            raise ValueError('Unknown action')
            