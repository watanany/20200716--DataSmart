#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from numpy.linalg import norm
from deap import base
from deap import tools

N = 4
SALES_CSV = pd.read_csv('./sales.csv')
TRADES_CSV = pd.read_csv('./trades.csv')
NAMES = list(pd.read_csv('./names.csv').get('顧客名'))

M = np.zeros((len(SALES_CSV), len(NAMES)))
for title, group in TRADES_CSV.groupby(['顧客名', '売り出し番号']):
    name, seq = title
    M[seq - 1][NAMES.index(name)] = len(group)

CXPB, MUTPB, NGEN = 0.5, 0.2, 1000

class Fitness(base.Fitness):
    def __init__(self, weights=(-1.0,), *args, **kwargs):
        self.weights = weights
        super().__init__(*args, **kwargs)

class Individual(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness = Fitness()

def create_toolbox():
    toolbox = base.Toolbox()
    toolbox.register('individual', tools.initRepeat, Individual, random.random, len(SALES_CSV) * N)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
    toolbox.register('select', tools.selTournament, tournsize=3)
    return toolbox

def update(toolbox, pop):
    children = toolbox.select(pop, len(pop))
    # XXX: clone need to evolve, but I don't know why this needs.
    children = [toolbox.clone(c) for c in children]

    mated_pairs = [(c1, c2) for c1, c2 in zip(children[::2], children[1::2]) if random.random() < CXPB]
    mutated_children = [c for c in children if random.random() < MUTPB]
    empty_children = [p[0] for p in mated_pairs] + [p[1] for p in mated_pairs] + mutated_children
    empty_children = { id(c):c for c in empty_children }.values()

    for c1, c2 in mated_pairs:
        toolbox.mate(c1, c2)
        del c1.fitness.values
        del c2.fitness.values
    for c in mutated_children:
        toolbox.mutate(c)
        del c.fitness.values
    for c in empty_children:
        c.fitness.values = toolbox.evaluate(c)

    return children, empty_children

def evaluate(individual):
    C = split_individual(individual)
    D = np.zeros((len(C), len(NAMES)))
    for j, v in enumerate(M.T):
        D[:, j] = np.array([norm(c - v) for c in C])
    return sum(d.min() for d in D.T),

def split_individual(individual):
    return np.split(individual, range(len(SALES_CSV), len(individual), len(SALES_CSV)))

def main():
    toolbox = create_toolbox()

    pop = toolbox.population(n=300)

    print('Start of evolution')
    print('  Evaluated %i individuals' % len(pop))

    for c in pop:
        c.fitness.values = toolbox.evaluate(c)

    for generation in range(NGEN):
        children, empty_children = update(toolbox, pop)
        pop[:] = children

        fits = np.array([c.fitness.values[0] for c in children])
        print('-- Generation %i --' % generation)
        print('  Evaluated {} individuals'.format(len(empty_children)))
        print('  Min %s' % fits.min())
        print('  Max %s' % fits.max())
        print('  Avg %s' % fits.mean())
        print('  Std %s' % np.std(fits))

    print('-- End of (successful) evolution --')

    best_child = tools.selBest(pop, 1)[0]
    for c in split_individual(best_child):
        print('Best individual is {}'.format(c))
    print('Best individual fitness values is {}'.format(best_child.fitness.values))

if __name__ == '__main__':
    main()
