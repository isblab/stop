import numpy as np
import random
from scipy.interpolate import RectBivariateSpline
import pickle
import sys
from scipy.stats import iqr
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import math

sys.path = ['../../'] + sys.path
from optimizer import ParameterGroup

random.seed(2021)


def ga_mutate(population, mutation_size, n):
    new_pop = []
    cnt = 0
    while cnt < n:
        g = population[np.random.choice(len(population))]
        new_pop.append(np.clip(np.array(g) + np.random.random((2,)) * 2 * mutation_size - mutation_size, -1, 1))
        cnt += 1
    return [x.tolist() for x in new_pop]


def ga_crossover(population, cross_over_rate):
    number_of_crossovers = math.ceil(cross_over_rate * len(population))
    number_of_crossovers -= (number_of_crossovers % 2)
    to_be_crossed = np.random.choice(list(range(len(population))), number_of_crossovers, replace=False)
    cross_over_pop = [population[x] for x in range(len(population)) if x not in to_be_crossed]
    to_be_crossed = to_be_crossed.tolist()
    random.shuffle(to_be_crossed)
    for i in range(len(to_be_crossed) // 2):
        j = -(i + 1)
        parent1 = population[i][:]
        parent2 = population[j][:]
        ind = np.random.choice([0, 1])
        parent1[ind] = population[j][ind]
        parent2[ind] = population[i][ind]
        cross_over_pop.append(parent1)
        cross_over_pop.append(parent2)
    return cross_over_pop


def ga_rank_fitness(foo, pop):
    fitness = [foo(i[0], i[1]) for i in pop]
    sorted_pop = sorted(zip(fitness, pop), reverse=True)
    return [x[1] for x in sorted_pop], np.array([x[0] for x in sorted_pop])


def ga_foo_wrapper(foo, target_range):
    def returned_foo(x, y):
        val = foo(x, y).flatten()[0]
        if val < target_range[0]:
            return val - target_range[0]
        elif val > target_range[1]:
            return target_range[1] - val
        else:
            return 0

    return returned_foo


def ga_2d(foo, initial_pop, mutation_size, target_range, retention_rate=0.25, cross_over_rate=0.5):
    pop = initial_pop
    foo = ga_foo_wrapper(foo, target_range)
    foo_evals = 0
    while foo_evals < 10000:
        pop, fitness = ga_rank_fitness(foo, pop)
        foo_evals += len(pop)
        if fitness[0] == 0:
            return True, foo_evals, [pop[x] for x in range(len(pop)) if fitness[x] == 0]
        retained_pop = pop[:int(retention_rate * len(pop))][:]
        new_pop = ga_mutate(pop[:int(retention_rate * len(pop))], mutation_size, len(pop) - len(retained_pop))
        pop = retained_pop + new_pop
        pop = ga_crossover(pop, cross_over_rate)
        random.shuffle(pop)
    return False, foo_evals, pop


np.random.seed(2021)


def normalize(x):
    start = -np.random.random() * 1.4 + 0.4
    end = np.random.random() * 1.4 + 0.6
    return (x - x.min()) / x.ptp() * (end - start) + start


multiple_foos = [[np.random.random((int(x), int(x))) for i in range(20)] for x in np.linspace(5, 25, 5)]
multiple_foos = [[normalize(x) for x in y] for y in multiple_foos]
multiple_foos = [[RectBivariateSpline(np.linspace(-1, 1, len(x)), np.linspace(-1, 1, len(x)), x, kx=3, ky=3) for x in y]
                 for y in multiple_foos]

with open('../examples/examples_2/optimization_data_5/logs/saved_optimizer_param_groups', 'rb') as f:
    opt_m5 = pickle.load(f)
fevals = []
for x in opt_m5[:20]:
    fevals.append(25 + 5 * (len([y for y in x.node_status if (y != 0)]) - 1))
cnt = len([x for x in opt_m5[:20] if 1 in x.node_status])
print(f'{cnt:^10}\t{np.median(fevals):^10.2f}\t{iqr(fevals):^10.2f}')

muts = np.linspace(0.02, 0.5, 20)

plt.figure()
h1, = plt.plot([np.min(muts), np.max(muts)], [np.median(fevals)] * 2, linestyle='-', color='black', label='median',
               lw=2)
h2, = plt.plot([np.min(muts), np.max(muts)], [np.percentile(fevals, 10)] * 2, linestyle='--', color='black',
               label='10th-percentile')
h3, = plt.plot([np.min(muts), np.max(muts)], [np.percentile(fevals, 90)] * 2, linestyle=':', color='black',
               label='90th-percentile')

ys = []
for mut_rate in muts:
    print(f'Mutation Rate {mut_rate}')
    for n_pop in [5, 50]:
        print(f'Population Size {n_pop}')
        subtemp = []
        cnt = 0
        for j in range(20):
            metric = multiple_foos[0][j]
            subsubtemp = []
            for k in range(10):
                random_pop = [(np.random.random((2,)) * 2 - 1).tolist() for i in range(n_pop)]
                vals = ga_2d(metric, random_pop, mut_rate, [0.48, 0.5])
                subsubtemp.append(vals[1])
                cnt += vals[0]
            subtemp += subsubtemp
        ys.append((np.median(subtemp), np.percentile(subtemp, 10), np.percentile(subtemp, 90), n_pop))
        print(f'{cnt:^10}\t{np.median(subtemp):^10.2f}\t{np.percentile(subtemp, 90):^10.2f}\t{np.max(subtemp):^10}')

plt.plot(muts, [x[0] for x in ys if x[3] == 5], color='red', lw=2)
plt.scatter(muts, [x[0] for x in ys if x[3] == 5], color='red')
c1 = [x[1] for x in ys if x[3] == 5]
c2 = [x[2] for x in ys if x[3] == 5]
plt.plot(muts, c1, linestyle='--', color='red')
plt.plot(muts, c2, linestyle=':', color='red')

plt.plot(muts, [x[0] for x in ys if x[3] == 50], color='green', lw=2)
plt.scatter(muts, [x[0] for x in ys if x[3] == 50], color='green')
c1 = [x[1] for x in ys if x[3] == 50]
c2 = [x[2] for x in ys if x[3] == 50]
plt.plot(muts, c1, linestyle='--', color='green')
plt.plot(muts, c2, linestyle=':', color='green')

temp = [Patch(facecolor='red', label='GA,population=5')]
temp += [Patch(facecolor='green', label='GA,population=50')]
temp += [Patch(facecolor='black', label='StOP')]

plt.legend(handles=temp + [h1, h2, h3], ncol=2)
plt.xlabel('Maximum Mutation Size')
plt.ylabel('Number of Function Evaluations')
plt.ylim(0, 200)
plt.xlim(np.min(muts), np.max(muts))
plt.savefig('../paper_example_figures/ga.svg')
plt.show()
