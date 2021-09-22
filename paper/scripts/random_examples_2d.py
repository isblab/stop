import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.cm as cm

sys.path = ['../../'] + sys.path
from optimizer import ParameterGroup

np.random.seed(2021)


def normalize(x):
    start = -np.random.random() * 1.4 + 0.4
    end = np.random.random() * 1.4 + 0.6
    return (x - x.min()) / x.ptp() * (end - start) + start


multiple_foos = [[np.random.random((int(x), int(x))) for i in range(20)] for x in np.linspace(5, 25, 5)]
multiple_foos = [[normalize(x) for x in y] for y in multiple_foos]
multiple_foos = [[RectBivariateSpline(np.linspace(-1, 1, len(x)), np.linspace(-1, 1, len(x)), x, kx=3, ky=3) for x in y]
                 for y in multiple_foos]

with open('../examples/examples_2/optimization_data_4/logs/saved_optimizer_param_groups', 'rb') as f:
    opt_m3 = pickle.load(f)

with open('../examples/examples_2/optimization_data_5/logs/saved_optimizer_param_groups', 'rb') as f:
    opt_m5 = pickle.load(f)

values_dict = dict()
for k, opt in enumerate([opt_m3, opt_m5]):
    for ii in range(5):
        for jj in range(20):
            obj = opt[ii * 20 + jj]
            fig, ax = plt.subplots()
            depths_taken = set()
            xvals = np.linspace(-1, 1, 200)
            zvals = multiple_foos[ii][jj](xvals, xvals)
            for i in range(len(obj.node_status)):
                if obj.node_status[i] == 0:
                    continue
                pd = obj.param_nodes[i]
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
                marker_list = ['o', 'v', 'P', '*', '1']
                d = obj.node_depth[i]
                if d in depths_taken:
                    lab = None
                else:
                    lab = f'Depth {d}'
                depths_taken.add(d)
                ax.scatter(p2, p1, c='black', marker=marker_list[d % len(marker_list)],
                           label=lab, s=30, zorder=10)

            all_vals = np.hstack([x[obj.metric_names[0]].flatten() for x in obj.metric_values_nodes])
            mx, mn = np.max(zvals.flatten()), np.min(zvals.flatten())
            rng = obj.metric_ranges[0]
            r1, r2 = (rng[0] - mn) / (mx - mn), (rng[1] - mn) / (mx - mn)
            r1, r2 = int(r1 * 256), int(r2 * 256)
            cmapnew = cmap(np.linspace(0, 1, 256))
            pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
            cmapnew[r1:r2 + 1, :] = pink
            cmapnew = ListedColormap(cmapnew)
            ax.set_ylabel(f'Parameter 1 Value')
            ax.set_xlabel(f'Parameter 2 Value')
            plt.pcolormesh(xvals, xvals, zvals, cmap=cmapnew, vmin=mn, vmax=mx, shading='gouraud',
                           zorder=-10)
            plt.colorbar()
            plt.title(f'm={[3, 5][k]}, curvyness={ii}, exampleno={jj}, status={obj.state}')
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.savefig(f'../random_metric_functions/fig_{ii}_{jj}_{k}_2d.png')
            plt.close('all')
            values_dict[(ii, jj, k)] = obj.state

for k in range(2):
    print(k, ':')
    for ii in range(5):
        z = [values_dict[i] for i in values_dict.keys() if (i[-1] == k) and (i[0] == ii)]
        print(f'\tCurvyness: {ii}, successes = {z.count(1)}/{len(z)}')