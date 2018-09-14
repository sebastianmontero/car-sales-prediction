from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import ray
from ray.tune import grid_search, run_experiments, register_trainable
from ensemble_evolver_trainable import EnsembleEvolverTrainable
                
ray.init()

register_trainable('ensemble_evolver_trainable', EnsembleEvolverTrainable)

run_experiments({
    'evolver_model_1n_weight_range_selection' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 40},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-model-20180815225523217235',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.5,
                'mutpb': 0.3,
                'num_best': grid_search([3, 10, 20]),
                'tournament_size': grid_search([3, 30, 60]),
                'population_size': 2000,
                'weight_range': grid_search([10,100,1000]),
                'zero_percentage': grid_search([10, 30, 50])
            },
            'repeat':1
        }
})


print('Experiment end')
          
        
        