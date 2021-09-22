import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path = ['../../'] + sys.path
from optimizer import ParameterGroup

np.random.seed(2021)


def get_good_regions(foo):
    x = np.linspace(-1, 1, 1000)
    y = foo(x)
    sign = np.sign(np.diff(y))
    sign_changes = np.diff(sign) != 0
    colors = []
    prev = 0
    for i in (np.where(sign_changes)[0]).tolist() + [len(y) - 1]:
        if any(np.logical_and(y[prev:i + 1] >= 0.48, y[prev:i + 1] <= 0.5)):
            colors.append('green')
        else:
            colors.append('red')
        prev = i + 1
    return (x[1:len(x) - 1][sign_changes]).tolist() + [1], colors


def normalize(x):
    start = -np.random.random() * 1.4 + 0.4
    end = np.random.random() * 1.4 + 0.6
    return (x - x.min()) / x.ptp() * (end - start) + start


multiple_foos = [[np.random.random(int(x)) for i in range(20)] for x in np.linspace(5, 25, 5)]
multiple_foos = [[normalize(x) for x in y] for y in multiple_foos]
multiple_foos = [[interp1d(np.linspace(-1, 1, len(x)), x, kind='cubic') for x in y] for y in multiple_foos]

with open('../examples/examples_2/optimization_data/logs/saved_optimizer_param_groups', 'rb') as f:
    opt_m_3 = pickle.load(f)

with open('../examples/examples_2/optimization_data_2/logs/saved_optimizer_param_groups', 'rb') as f:
    opt_m_5 = pickle.load(f)

with open('../examples/examples_2/optimization_data_3/logs/saved_optimizer_param_groups', 'rb') as f:
    opt_m_7 = pickle.load(f)

values_dict = dict()
for k, opt in enumerate([opt_m_3, opt_m_5, opt_m_7]):
    for ii in range(5):
        for jj in range(20):
            obj = opt[ii * 20 + jj]
            plt.figure()
            plt.plot(np.linspace(-1, 1, 250), multiple_foos[ii][jj](np.linspace(-1, 1, 250)), linestyle='--',
                     color='black', alpha=0.5)
            cmap_depth = ['green', 'red', 'blue', 'orange', 'olive']
            for i in range(len(obj.node_status)):
                if obj.node_status[i] == 0:
                    continue
                pd = obj.param_nodes[i]
                p = pd[list(pd.keys())[0]]
                mvals = [obj.metric_values_nodes[i][obj.metric_names[0]]]
                if i > 0:
                    p = p[1:len(p) - 1]
                    mvals = [x[1:len(x) - 1] for x in mvals]
                d = obj.node_depth[i]
                lab = f'Depth {d}'
                plt.scatter(p, mvals[0], label=lab, color=cmap_depth[d % 10], marker='^', s=40,
                            zorder=10)
            rng = [0.48, 0.5]
            plt.fill_between([-1, 1], [rng[0], rng[0]], [rng[1], rng[1]], color='none', hatch='///',
                             edgecolor='gray',
                             zorder=-20)
            plt.xlabel('Parameter Value')
            plt.ylabel('Metric Value')
            plt.title(f'm={[3, 5, 7][k]}, curvyness={ii}, exampleno={jj}, status={obj.state}')
            plt.savefig(f'../random_metric_functions/fig_{ii}_{jj}_{k}.png')
            # sign_indices, cols = get_good_regions(multiple_foos[ii][jj])
            # ymin, ymax = plt.ylim()
            # prev = -1
            # for i in range(len(sign_indices)):
            #     plt.fill_between([prev, sign_indices[i]], [ymin, ymin], [ymax, ymax], color=cols[i], alpha=0.1)
            #     prev = sign_indices[i]
            # plt.savefig(f'../random_metric_functions/fig_{ii}_{jj}_{k}_colorized.png')
            plt.close('all')
            values_dict[(ii, jj, k)] = obj.state

for k in range(3):
    print(k, ':')
    for ii in range(5):
        z = [values_dict[i] for i in values_dict.keys() if (i[-1] == k) and (i[0] == ii)]
        print(f'\tCurvyness: {ii}, successes = {z.count(1)}/{len(z)}')
