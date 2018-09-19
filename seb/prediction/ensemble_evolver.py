'''
Created on Sep 13, 2018

@author: nishilab
'''

import random
import numpy as np
from ensemble_evaluator import InvalidEnsembleWeights
from deap import base, creator, tools


class EnsembleEvolver(object):
    
    def __init__(self, config, ensemble_evaluator):
        self._ensemble_evaluator = ensemble_evaluator
        self._setup(config)
        self._init_population()
        
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
        
        max_ = config['weight_range']
        min_ =  -1 * (max_ * config['zero_percentage'])//100
        toolbox.register('rand_int', random.randint, min_, max_)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.rand_int, self._ensemble_evaluator._num_networks)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        def evaluate_individual(ind):
            try:
                return (self._ensemble_evaluator.test_ensemble(ind),)
            except InvalidEnsembleWeights:
                return (10000,)
            
        toolbox.register('evaluate', evaluate_individual)
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', tools.mutUniformInt, low=min_, up=max_, indpb=config['indpb'])
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
    
    def _init_population(self):
        self._generation = 0
        self._population = self._toolbox.population(n=self._config['population_size'])
        self.evaluate(self._population)
       
    @property
    def generation(self):
        return self._generation 
    
    @property
    def logbook(self):
        return self._logbook 
        
    def evolve(self):
        for _ in range(self._config['num_generations']):
            best_ind, best_fitness = self.evolve_step(1)
        
        print('Best individual {}%:'.format(best_fitness))
        print(best_ind)
        
        return best_ind
    
    def evolve_step(self, gens):
        
        config = self._config
        toolbox = self._toolbox
        num_best = config['num_best']
        pop = self._population
        
        for _ in range(gens):
            
            record = self._stats.compile(pop)
            self._logbook.record(gen=self._generation, **record)
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
            self._generation += 1
        
        best_ind = self._get_best_ind()
        return best_ind, best_ind.fitness.values[0]
    
    def get_best_ensemble(self):
        self._ensemble_evaluator.weights = self._get_best_ind()
        return self._ensemble_evaluator
    
    def _get_best_ind(self):
        return self._toolbox.select_best(self._population, 1)[0]
        
            
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
        