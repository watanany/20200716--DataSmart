#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import random
from numpy.linalg import norm
from deap import base
from deap import creator
from deap import tools

sns.set(color_codes=True)

N = 4
g_sales = pd.read_csv('./sales.csv')
g_trades = pd.read_csv('./trades.csv')
g_names = list(pd.read_csv('./names.csv').get('顧客名'))

M = np.zeros((len(g_sales), len(g_names)))
for title, group in g_trades.groupby(['顧客名', '売り出し番号']):
    name, seq = title
    M[seq - 1][g_names.index(name)] = len(group)

CXPB, MUTPB, NGEN = 0.5, 0.2, 1000

def evaluate(individual):
    C = np.split(individual, range(len(g_sales), len(individual), len(g_sales)))
    D = np.zeros((len(C), len(g_names)))
    for j, v in enumerate(M.T):
        D[:, j] = np.array([norm(c - v) for c in C])
    return sum(d.min() for d in D.T),

def init_creator():
    creator.create('Fitness', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.Fitness)

def create_toolbox():
    toolbox = base.Toolbox()
    toolbox.register('individual', tools.initRepeat, creator.Individual, random.random, len(g_sales) * N)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
    toolbox.register('select', tools.selTournament, tournsize=3)
    return toolbox

def main():
    init_creator()
    toolbox = create_toolbox()

    pop = toolbox.population(n=300)

    print('Start of evolution')

    for c in pop:
        c.fitness.values = toolbox.evaluate(c)

    print('  Evaluated %i individuals' % len(pop))

    for g in range(NGEN):
        print('-- Generation %i --' % g)
        children = toolbox.select(pop, len(pop))

        # XXX: clone need to evolve, but I don't know why this needs.
        children = [toolbox.clone(c) for c in children]

        mated_pairs = [(c1, c2) for c1, c2 in zip(children[::2], children[1::2]) if random.random() < CXPB]
        mutated_children = [c for c in children if random.random() < MUTPB]

        for c1, c2 in mated_pairs:
            toolbox.mate(c1, c2)
            del c1.fitness.values
            del c2.fitness.values
        for c in mutated_children:
            toolbox.mutate(c)
            del c.fitness.values

        empty_children = [c for c in children if not c.fitness.valid]
        for c in empty_children:
            c.fitness.values = toolbox.evaluate(c)

        print('  Evaluated {} individuals'.format(len(empty_children)))
        pop[:] = children

        fits = np.array([c.fitness.values[0] for c in children])

        print('  Min %s' % fits.min())
        print('  Max %s' % fits.max())
        print('  Avg %s' % fits.mean())
        print('  Std %s' % np.std(fits))

    print('-- End of (successful) evolution --')

    best_child = tools.selBest(pop, 1)[0]
    for c in np.split(best_child, range(len(g_sales), len(best_child), len(g_sales))):
        print('Best individual is {}'.format(c))
    print('Best individual fitness values is {}'.format(best_child.fitness.values))

if __name__ == '__main__':
    main()
