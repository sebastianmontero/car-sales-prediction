'''
Created on Sep 13, 2018

@author: nishilab
'''

import random
import numpy as np
from deap import base, creator, tools


class EnsembleEvolver(object):
    
    def __init__(self, config, ensemble_evaluator):
        self._config = config
        self._ensemble_evaluator = ensemble_evaluator
        self._setup(config)
        
        
    def _setup(self, config):
        
        creator.create('FitnessMin', base.Fitness, weights = (-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        toolbox.register('rand_float', random.uniform, 0.0, 1.0)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.rand_float, self._ensemble_evaluator._num_networks)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        toolbox.register('evaluate', lambda ind: self._ensemble_evaluator.test_ensemble(ind))
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', SebsToolbox.mutUniformFloat, indpb=config['indpb'])
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
        
        pop = toolbox.population(n=config['population_size'])
        
        print("Start of evolution")
        
        self.evaluate(pop)
        
        for gen in range(config['num_generations']):
            
            record = self._stats.compile(pop)
            self._logbook.record(generation=gen, **record)
            print(self._logbook.stream)
            
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config['CXPB']:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < config['MUTPB']:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                    
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.evaluate(invalid_ind)
            
            pop[:] = offspring
            
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
        