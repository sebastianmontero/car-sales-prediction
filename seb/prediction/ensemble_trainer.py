from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import ray
from ray.tune import grid_search, run_experiments, register_trainable

from model_trainable import ModelTrainable
from ensemble_config import EnsembleConfig


class EnsembleTrainer():
     
    
    def __init__(self, config, repeats, description=''):        
        self._repeats = repeats 
        self._config = config
        config['store_window'] = False
        self._config = config
        self._ensembleConfig = EnsembleConfig(description=description, base_path=ModelTrainable.BASE_PATH)
        self._ray_results_dir = os.path.join(os.path.expanduser('~'), 'ray_results', self._ensembleConfig.get_ensemble_base_dir())
         
    
    def run(self):
        
        experiment_name = self._ensembleConfig.get_ensemble_base_dir() 
            
        run_experiments({
            experiment_name : {
                'run': 'car_sales_prediction_trainable',
                'trial_resources': {'cpu': 8, 'gpu': 1},
                'stop': {'training_iteration': 60},
                'config' : self._config,
                'repeat':self._repeats,
                'local_dir': self._ray_results_dir
            }
        })  
        print('Finished ensemble training!')
    

ray.init()
register_trainable('car_sales_prediction_trainable', ModelTrainable)

ensemble_trainer = EnsembleTrainer({
                'line_id': 102,
                'keep_prob' : 1,
                'layers' : [15],
                'max_epoch' : 2,
                'window_size': 37
            }, repeats=3)
          
ensemble_trainer.run()


