'''
Created on Sep 13, 2018

@author: nishilab
'''

import random
import numpy as np
from deap import base, creator, tools


class EnsembleEvolver(object):
    
    def __init__(self, config, ensemble_evaluator):
        self._ensemble_evaluator = ensemble_evaluator
        self._setup(config)
        
        
    def _setup(self, config):
        
        self._config = {
            'indpb': 0.05,
            'cxpb': 0.5,
            'mutpb': 0.3,
            'num_best': 3,
            'num_generations': 20000,
            'tournament_size': 3,
            'population_size': 500,
            'weight_range': 1000,
            'zero_percentage': 20
        }
        
        self._config.update(config)
        config = self._config
        creator.create('FitnessMin', base.Fitness, weights = (-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        max = config['weight_range']
        min =  -1 * (max * config['zero_percentage'])//100
        toolbox.register('rand_int', random.randint, min, max)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.rand_int, self._ensemble_evaluator._num_networks)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        toolbox.register('evaluate', lambda ind: (self._ensemble_evaluator.test_ensemble(ind),))
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', tools.mutUniformInt, low=min, up=max, indpb=config['indpb'])
        toolbox.register('select_best', tools.selBest)
        toolbox.register('select', tools.selTournament, tournsize=config['tournament_size'])
        
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)
        
        self._toolbox = toolbox
        self._stats = stats
        self._logbook = tools.Logbook()
        self._logbook.header = 'gen', 'min', 'max', 'avg', 'std'
        
    def evolve(self):
        config = self._config
        toolbox = self._toolbox
        num_best = config['num_best']
        
        pop = toolbox.population(n=config['population_size'])
        
        print("Start of evolution")
        
        self.evaluate(pop)
        
        for gen in range(config['num_generations']):
            
            record = self._stats.compile(pop)
            self._logbook.record(gen=gen, **record)
            print(self._logbook.stream)
            
            best = toolbox.select_best(pop, num_best)
            offspring = toolbox.select(pop, len(pop) - num_best)
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config['cxpb']:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < config['mutpb']:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                    
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.evaluate(invalid_ind)
            
            pop[:num_best] = best
            pop[num_best:] = offspring
        
        best_ind = toolbox.select_best(pop, 1)[0]
        print('Best individual {}%:'.format(best_ind.fitness.values))
        print(best_ind)
        
        return best_ind
            
    def evaluate(self, population):
        fitnesses = map(self._toolbox.evaluate, population)
        
        for ind, fitness in zip(population, fitnesses):
            ind.fitness.values = fitness 
            
class SebsToolbox():
    
    @classmethod
    def mutUniformFloat(cls, individual, indpb, min_=0.0, max_=1.0):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(min_, max_)
        return individual,
        