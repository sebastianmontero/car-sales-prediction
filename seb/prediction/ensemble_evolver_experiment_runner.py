from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import ray
from ray.tune import grid_search, run_experiments, register_trainable
from ensemble_evolver_trainable import EnsembleEvolverTrainable
                
ray.init()

register_trainable('ensemble_evolver_trainable', EnsembleEvolverTrainable)


run_experiments({
    'evolver_model_2n_r5_focus' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 70},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-model_2n_r5-20180920234425217425',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.7,
                'mutpb':  0.5,
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 5000,
                'weight_range': 100,
                'zero_percentage': 50
            },
            'repeat':5
    }
})

'''run_experiments({
    'evolver_platform_1n_focus' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 70},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-platform_23_37-20180826000513927940',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.7,
                'mutpb':  0.5,
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 5000,
                'weight_range': 100,
                'zero_percentage': 50
            },
            'repeat':5
    }
})'''
'''run_experiments({
    'evolver_platform_3n_r4_focus' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 70},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-platform_3n_r4-20180915170334129906',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.7,
                'mutpb':  0.5,
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 5000,
                'weight_range': 100,
                'zero_percentage': 50
            },
            'repeat':5
    },
    'evolver_nationwide_1n_focus' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 70},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-nationwide-20180814214839162428',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.7,
                'mutpb':  0.5,
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 5000,
                'weight_range': 100,
                'zero_percentage': 50
            },
            'repeat':5
    },
    'evolver_nationwide_2n_focus' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 70},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-nationwide_2n-20180821215754041110',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.7,
                'mutpb':  0.5,
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 5000,
                'weight_range': 100,
                'zero_percentage': 50
            },
            'repeat':5
    },
    'evolver_nationwide_3n_focus' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 70},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-nationwide_3n-20180905085329121812',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.7,
                'mutpb':  0.5,
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 5000,
                'weight_range': 100,
                'zero_percentage': 50
            },
            'repeat':5
    }
})'''
'''run_experiments({
    'evolver_model_3n_r4_focus' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 70},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-model_3n-20180908235408987343',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.7,
                'mutpb':  0.5,
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 5000,
                'weight_range': 100,
                'zero_percentage': 50
            },
            'repeat':5
    },
    'evolver_platform_2n_r4_focus' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 70},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-platform_2n_r4-20180914223820934330',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.7,
                'mutpb':  0.5,
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 5000,
                'weight_range': 100,
                'zero_percentage': 50
            },
            'repeat':5
    }
})'''

'''run_experiments({
    'evolver_model_3n_r4_focus' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 70},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-model_3n-20180908235408987343',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.7,
                'mutpb':  0.5,
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 5000,
                'weight_range': 100,
                'zero_percentage': 50
            },
            'repeat':5
        }
})'''

'''run_experiments({
    'evolver_model_1n_r4_focus' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 70},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-model_1n_r4-20180916111403290168',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.7,
                'mutpb':  0.5,
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 5000,
                'weight_range': 100,
                'zero_percentage': 50
            },
            'repeat':5
        }
})'''

'''run_experiments({
    'evolver_model_1n_probabilites_finer' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 40},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-model-20180815225523217235',
                'gens_per_step' : 10,
                'indpb': grid_search([0.01, 0.03, 0.05, 0.07]),
                'cxpb': grid_search([0.4, 0.6, 0.8]),
                'mutpb': grid_search([0.4, 0.5, 0.6, 0.7]),
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 3000,
                'weight_range': 50,
                'zero_percentage': 50
            },
            'repeat':1
        }
})'''

'''run_experiments({
    'evolver_model_1n_probabilites' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 40},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-model-20180815225523217235',
                'gens_per_step' : 10,
                'indpb': grid_search([0.01, 0.05, 0.1, 0.2]),
                'cxpb': grid_search([0.2, 0.5, 0.8]),
                'mutpb': grid_search([0.1, 0.3, 0.6]),
                'num_best': 20,
                'tournament_size': 80,
                'population_size': 3000,
                'weight_range': 50,
                'zero_percentage': 50
            },
            'repeat':1
        }
})'''

'''run_experiments({
    'evolver_model_1n_weight_range_selection_finer' : {
            'run': 'ensemble_evolver_trainable',
            'trial_resources': {'cpu': 8, 'gpu': 0},
            'stop': {'training_iteration': 40},
            'config' : {
                'run_path': '/home/nishilab/Documents/python/model-storage/ensemble-run-model-20180815225523217235',
                'gens_per_step' : 10,
                'indpb': 0.05,
                'cxpb': 0.5,
                'mutpb': 0.3,
                'num_best': grid_search([10, 20, 30]),
                'tournament_size': grid_search([10, 40, 80]),
                'population_size': 3000,
                'weight_range': grid_search([50,100,300]),
                'zero_percentage': grid_search([50, 60])
            },
            'repeat':1
        }
})'''

'''run_experiments({
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
                'population_size': 3000,
                'weight_range': grid_search([10,100,1000]),
                'zero_percentage': grid_search([10, 30, 50])
            },
            'repeat':1
        }
})'''


print('Experiment end')
          
        
        