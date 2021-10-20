import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pickle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize
import sys

sys.path = ['../../'] + sys.path
from optimizer import ParameterGroup


def foo(x):
    return 1 - x ** 2


def foo2(x):
    return x * 0.5 + 0.7


def foo3(x):
    return 1 - x ** 3 - 1.2 * x ** 2 + 0.5 * x


def foo4(x):
    return 1 - (x - 0.5) ** 2


def plot_for_2d(self):
    fig, ax = plt.subplots()
    depths_taken = set()
    for i in range(len(self.node_status)):
        if self.node_status[i] == 0:
            continue
        pd = self.param_nodes[i]
        p1 = pd[list(pd.keys())[0]]
        p2 = pd[list(pd.keys())[1]]
        if i > 0:
            if len(p1) == 1:
                p2 = p2[1:len(p2) - 1]
                p1 = np.ones(len(p2)) * p1[0]
            else:
                p1 = p1[1:len(p1) - 1]
                p2 = np.ones(len(p1)) * p2[0]
        else:
            p1 = np.tile(p1[:, np.newaxis], (1, len(p1))).flatten()
            p2 = np.tile(p2, (len(p2), 1)).flatten()
        cmap = cm.get_cmap('viridis', 256)
        marker_list = ['o', 'v', '^', '<', '>', 'P', '*', '1', '2', '3', '4']
        d = self.node_depth[i]

        if d in depths_taken:
            lab = None
        else:
            lab = f'Depth {d}'
        depths_taken.add(d)
        ax.scatter(p1 / 2, p2 / 3, c='black', marker=marker_list[d % len(marker_list)],
                   label=lab, s=60, zorder=10)

    ax.set_title('')
    ax.set_ylabel(f'Parameter 1 Value')
    ax.set_xlabel(f'Parameter 2 Value')
    ax.legend()
    xvals = np.linspace(-1.1, 1.1, 100)
    xvals, yvals = np.meshgrid(xvals, xvals)
    zvals = np.clip(1 - ((xvals + yvals) / 2) ** 2, 0, 1)
    mx, mn = 1, 0
    rng = (0.6, 0.68)
    r1, r2 = (rng[0] - mn) / mx, (rng[1] - mn) / mx
    r1, r2 = int(r1 * 256), int(r2 * 256)
    cmapnew = cmap(np.linspace(0, 1, 256))
    pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
    cmapnew[r1:r2 + 1, :] = pink
    cmapnew = ListedColormap(cmapnew)
    norm = Normalize(vmin=mn, vmax=mx)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmapnew),
                 orientation='vertical')
    plt.pcolormesh(np.linspace(-1.1, 1.1, 100), np.linspace(-1.1, 1.1, 100), zvals, cmap=cmapnew, shading='nearest',
                   zorder=-10)
    plt.savefig('../paper_example_figures/fig_2d_0.svg')
    plt.close('all')


xvals = np.linspace(-1, 1, 100)
yvals = foo(xvals)
yvals2 = foo2(xvals)
yvals3 = foo3(xvals)
yvals4 = foo4(xvals)

with open('../examples/examples_1/optimization_data/logs/saved_optimizer_param_groups', 'rb') as f:
    obj_list = pickle.load(f)
yval_biglist = [(yvals,), (yvals, yvals2), (yvals, yvals3), (yvals4,)]
xval_biglist = [1, 4, 5, 6]
ind = -1
for obj in obj_list:
    if len(obj.parameter_names) > 1:
        plot_for_2d(obj)
        continue
    ind += 1
    if ind == len(yval_biglist):
        break
    y = yval_biglist[ind]
    xv = xval_biglist[ind]
    fig, ax = plt.subplots()
    depths_taken = set()
    cmap_depth = ['green', 'red', 'blue', 'orange']
    markers = ['o', 'x']
    linestyles = ['--', ':']
    for i in range(len(obj.node_status)):
        if obj.node_status[i] == 0:
            continue
        pd = obj.param_nodes[i]
        p = pd[list(pd.keys())[0]]
        mvals = []
        mvals_sd = []
        for m in obj.metric_names:
            mvals.append(obj.metric_values_nodes[i][m])
            mvals_sd.append(obj.metric_values_nodes_sd[i][m])
        if i > 0:
            p = p[1:len(p) - 1]
            mvals = [x[1:len(x) - 1] for x in mvals]
            mvals_sd = [x[1:len(x) - 1] for x in mvals_sd]
        d = obj.node_depth[i]
        for j in range(len(obj.metric_names)):
            if d in depths_taken:
                lab = None
            else:
                lab = f'Depth {d}'
            depths_taken.add(d)
            ax.scatter(p / xv, mvals[j], label=lab, color=cmap_depth[d % 10], marker=markers[j], s=40, zorder=10)
            ax.errorbar(p / xv, mvals[j], yerr=mvals_sd[j], fmt='none', ecolor=cmap_depth[d % 10], capsize=2,
                        zorder=-10)
            ax.plot(xvals, y[j], color='black', linestyle=linestyles[j])
    c = ['///', '...']
    x1, x2 = ax.get_xlim()
    for i in range(len(obj.metric_names)):
        rng = obj.metric_ranges[i]
        ax.fill_between([x1, x2], [rng[0], rng[0]], [rng[1], rng[1]], color='none', hatch=c[i], edgecolor='gray',
                        zorder=-20)
    legend_elements = [Line2D([0], [0], marker=markers[0], linestyle=linestyles[0], color='black', label='Metric 1'),
                       Line2D([0], [0], marker=markers[1], linestyle=linestyles[1], color='black', label='Metric 2')]
    if len(obj.metric_names) == 1:
        del legend_elements[-1]
    legend_elements += [Patch(facecolor='none', edgecolor='gray', hatch=c[0], label='Metric 1 target'),
                        Patch(facecolor='none', edgecolor='gray', hatch=c[1], label='Metric 2 target')]
    if len(obj.metric_names) == 1:
        del legend_elements[-1]
    depth_cols = [(list(depths_taken)[i], cmap_depth[i % 10]) for i in range(len(depths_taken))]
    temp = [Patch(facecolor=col, label=f'Depth={dr}') for dr, col in depth_cols]
    legend_elements += temp
    ax.legend(handles=legend_elements)
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Metric Value')
    ax.set_title('')
    plt.savefig(f'../paper_example_figures/fig_{ind}.svg')
    plt.close('all')
