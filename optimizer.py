import numpy as np
import os
import time
import warnings
import re
import shlex
import copy
import matplotlib
from shutil import rmtree
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from itertools import product
from utils import iterate_over_all_axes, Job
from tqdm import tqdm

matplotlib.use('agg')


# Contains all the details of a single parameter group (methods allow updating the DFS-tree for this group)
class ParameterGroup:

    @classmethod
    def create_groups(cls, parameter_names, parameter_groups, parameter_ranges, metric_names, metric_groups,
                      metric_ranges, metric_search_strings, max_depth):
        groups = []
        for i in range(max(parameter_groups) + 1):
            p_group = [parameter_names[j] for j in range(len(parameter_names)) if parameter_groups[j] == i]
            m_group = [metric_names[j] for j in range(len(metric_names)) if metric_groups[j] == i]
            p_ranges = [parameter_ranges[j] for j in range(len(parameter_names)) if parameter_groups[j] == i]
            m_ranges = [metric_ranges[j] for j in range(len(metric_names)) if metric_groups[j] == i]
            m_strs = [metric_search_strings[j] for j in range(len(metric_names)) if metric_groups[j] == i]
            groups.append(cls(p_group, p_ranges, m_group, m_ranges, i, m_strs, max_depth))
        return groups

    def __init__(self, parameter_names, parameter_ranges, metric_names, metric_ranges, group_id,
                 metric_search_strings, max_depth):
        self.parameter_names = parameter_names
        self.metric_names = metric_names
        self.initial_parameter_ranges = parameter_ranges
        self.metric_ranges = metric_ranges
        self.metric_search_strings = metric_search_strings
        self.optimized_values = dict()  # Key: parameter name, Value: Array of optimized values
        self.optimized_metric_values = dict()  # Key: Metric name, Value: Metric value corresponding to above
        self.last_used_values = dict()  # Key: parameter name, Value: Array of last loaded values
        # Each node is a dictionary with Key: Parameter name, Value: Array of values to evaluate at this node
        # The nD vs 1D nature of the node is inferred from the dictionary (i.e. the length of array for diff parameters)
        # Node with all parameters having m-length Array (m > 1): nD grid search
        # Node with only one parameter having m-length Array (m > 1) and the rest 1-length array: 1D search
        self.param_nodes = []
        # Each node is a dictionary with Key: Metric name, Value: nD-Array of values (mean of n_repeat runs)
        # In the nD grid, the ith dimension represents the ith parameter; each point is a combination of all params
        self.metric_values_nodes = []
        # Same as above but for sd of the n_repeat runs
        self.metric_values_nodes_sd = []
        # -1: Failed, 0: Yet to run/currently running, 1: Successful
        self.node_status = []
        self.node_depth = []
        self.max_depth = int(max_depth)
        self.current_node = None
        self.node_parents = []
        self.node_children = []
        # The run_number of the overall optimization run corresponding to each point in the node
        # Each IMP run is identified as {run_number}_{repeat_number}
        # While the run_numbers are not assigned, NaNs are added to make it easier to work with
        self.node_run_numbers = []
        self.group_id = group_id
        self.state = 0  # -1: Failed, 0: Yet to run/currently running, 1: Successful
        self.run_number = 0  # To increment and keep track of the current overall run number
        self.issues = set()  # To print in the report

    def dfs_continue(self):  # Return True if continuation is possible, False otherwise
        # Updates the current_node if continuation is possible
        for i in self.node_children[self.current_node]:
            if self.node_status[i] == 0:
                self.current_node = i
                return True
        if self.node_parents[self.current_node] == self.current_node:
            return False
        self.current_node = self.node_parents[self.current_node]
        return self.dfs_continue()

    # Returns the next set of runs needed to run the current_node
    def get_next_block(self):
        # Root node needs all the runs to be initiated anew
        # All other nodes have two points that are already run from the parent node
        pd = self.param_nodes[self.current_node]
        n_params = len(self.parameter_names)
        if n_params == 1:
            z = copy.deepcopy(pd)
            if self.node_parents[self.current_node] != self.current_node:  # Not the root node
                all_vals = z[self.parameter_names[0]]
                # Skip the first and the last (as they are already run at the parent node)
                new_vals = all_vals[1:len(all_vals) - 1]
                z[self.parameter_names[0]] = new_vals
                missing = np.isnan(self.node_run_numbers[self.current_node])
                m_1d = np.sum(missing)
                self.node_run_numbers[self.current_node][missing] = np.arange(self.run_number, self.run_number + m_1d)
                self.node_run_numbers[self.current_node] = np.array(self.node_run_numbers[self.current_node],
                                                                    dtype=np.uint16)
                self.run_number += m_1d
            # The root node case is handled in initial_setup
        elif self.node_parents[self.current_node] != self.current_node:  # not the root node -> only a 1D search
            param_to_vary = [x for x in pd.keys() if len(pd[x]) > 1]  # only the parameter to vary for nD param groups
            assert len(param_to_vary) == 1
            param_to_vary = param_to_vary[0]
            z = dict()
            z[param_to_vary] = pd[param_to_vary][1:len(pd[param_to_vary]) - 1]
            for j in pd.keys():
                if j == param_to_vary:
                    continue
                # For all the other params (not varying in the runs of this block), repeat the fixed value m times
                z[j] = np.ones(pd[param_to_vary].shape[0] - 2) * pd[j][0]
            missing = np.isnan(self.node_run_numbers[self.current_node])
            m_1d = np.sum(missing)
            self.node_run_numbers[self.current_node][missing] = np.arange(self.run_number, self.run_number + m_1d)
            self.node_run_numbers[self.current_node] = np.array(self.node_run_numbers[self.current_node],
                                                                dtype=np.uint16)
            self.run_number += m_1d
        else:  # nD grid search (root node)
            new_size = len(pd[self.parameter_names[0]]) ** n_params
            z = dict()
            for j in self.parameter_names:
                z[j] = np.zeros(new_size)
            order = product(np.arange(len(pd[self.parameter_names[0]])), repeat=n_params)
            index = 0
            for i in order:
                for j in range(n_params):
                    z[self.parameter_names[j]][index] = pd[self.parameter_names[j]][i[j]]
                index += 1
        self.last_used_values = z
        return copy.deepcopy(z)

    def initial_setup(self, m_nd, m_1d):
        # Handle the root node setup
        n_params = len(self.parameter_names)
        if n_params > 1:
            m = m_nd[n_params - 2]
        else:
            m = m_1d
        n_metrics = len(self.metric_names)
        param_1d_arrays = dict()
        for j in range(n_params):
            lower_lim, upper_lim = self.initial_parameter_ranges[j]
            param_1d_arrays[self.parameter_names[j]] = np.linspace(lower_lim, upper_lim, m, dtype=np.float32)
        self.param_nodes.append(param_1d_arrays)
        m_nd_vals = dict()
        for j in self.metric_names:
            m_nd_vals[j] = np.zeros(tuple(m for i in range(n_params)), dtype=np.float32)
            m_nd_vals[j][:] = np.NaN
        self.metric_values_nodes.append(m_nd_vals)
        self.metric_values_nodes_sd.append(copy.deepcopy(m_nd_vals))
        self.node_status.append(0)
        self.current_node = 0
        self.node_parents.append(0)
        self.node_children.append([])
        self.node_depth.append(0)
        n_runs = m ** n_params
        self.node_run_numbers.append(np.arange(self.run_number, self.run_number + n_runs, dtype=np.uint16))
        self.run_number += n_runs
        return n_params, n_metrics, n_runs

    # Plots the metric landscape as per the sampling
    def plot_progress(self, location):
        if len(self.parameter_names) == 2:  # Needs a 2D plot
            figs, axs = [], []
            depths_taken = []  # To prevent overburdening the legend
            for m in range(len(self.metric_names)):
                fig, ax = plt.subplots()
                figs.append(fig)
                axs.append(ax)
                depths_taken.append(set())
            for i in range(len(self.node_status)):
                if self.node_status[i] == 0:
                    continue
                pd = self.param_nodes[i]
                p1 = pd[list(pd.keys())[0]]
                p2 = pd[list(pd.keys())[1]]
                mvals = []
                for m in self.metric_names:
                    mvals.append(self.metric_values_nodes[i][m])
                if i > 0:
                    if len(p1) == 1:
                        p2 = p2[1:len(p2) - 1]
                        p1 = np.ones(len(p2)) * p1[0]
                    else:
                        p1 = p1[1:len(p1) - 1]
                        p2 = np.ones(len(p1)) * p2[0]
                    mvals = [x[1:len(x) - 1] for x in mvals]
                else:
                    p1 = np.tile(p1[:, np.newaxis], (1, len(p1))).flatten()
                    p2 = np.tile(p2, (len(p2), 1)).flatten()
                    mvals = [x.flatten() for x in mvals]
                cmap = cm.get_cmap('viridis', 256)
                marker_list = ['o', 'v', '^', '<', '>', 'P', '*', '1', '2', '3', '4']
                d = self.node_depth[i]
                for j in range(len(self.metric_names)):
                    all_vals = np.hstack([x[self.metric_names[j]].flatten() for x in self.metric_values_nodes])
                    mx, mn = np.nanmax(all_vals), np.nanmin(all_vals)
                    rng = self.metric_ranges[j]
                    mx = max(mx, max(rng))
                    mn = min(mn, min(rng))
                    r1, r2 = (rng[0] - mn) / (mx - mn), (rng[1] - mn) / (mx - mn)
                    r1, r2 = int(r1 * 256), int(r2 * 256)
                    cmapnew = cmap(np.linspace(0, 1, 256))
                    pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
                    cmapnew[r1:r2 + 1, :] = pink
                    cmapnew = ListedColormap(cmapnew)
                    if d in depths_taken[j]:
                        lab = None
                    else:
                        lab = f'Depth {d}'
                    depths_taken[j].add(d)
                    axs[j].scatter(p1, p2, c=cmapnew((mvals[j] - mn) / (mx - mn)), marker=marker_list[d % len(marker_list)],
                                   label=lab, s=15, zorder=10)
            for i in range(len(self.metric_names)):
                all_vals = np.hstack([x[self.metric_names[i]].flatten() for x in self.metric_values_nodes])
                mx, mn = np.nanmax(all_vals), np.nanmin(all_vals)
                rng = self.metric_ranges[i]
                mx = max(mx, max(rng))
                mn = min(mn, min(rng))
                r1, r2 = (rng[0] - mn) / (mx - mn), (rng[1] - mn) / (mx - mn)
                r1, r2 = int(r1 * 256), int(r2 * 256)
                cmapnew = cmap(np.linspace(0, 1, 256))
                pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
                cmapnew[r1:r2 + 1, :] = pink
                cmapnew = ListedColormap(cmapnew)
                norm = Normalize(vmin=mn, vmax=mx)
                name = self.metric_names[i]
                figs[i].colorbar(cm.ScalarMappable(norm=norm, cmap=cmapnew),
                                 orientation='vertical')
                axs[i].set_title(f'Overall Progress for {name}')
                axs[i].set_ylabel(f'{self.parameter_names[1]}')
                axs[i].set_xlabel(f'{self.parameter_names[0]}')
                axs[i].legend()
                figs[i].savefig(f'{location}/{self.group_id}_{name}_overall_progress.png')
                plt.close(figs[i])
            plt.close('all')
        if len(self.parameter_names) == 1:
            figs, axs = [], []
            depths_taken = []
            for i in range(len(self.metric_names)):
                fig, ax = plt.subplots()
                figs.append(fig)
                axs.append(ax)
                depths_taken.append(set())
            cmap_depth = cm.get_cmap('tab10', 10)(np.linspace(0, 1, 10))
            for i in range(len(self.node_status)):
                if self.node_status[i] == 0:
                    continue
                pd = self.param_nodes[i]
                p = pd[list(pd.keys())[0]]
                mvals = []
                mvals_sd = []
                for m in self.metric_names:
                    mvals.append(self.metric_values_nodes[i][m])
                    mvals_sd.append(self.metric_values_nodes_sd[i][m])
                if i > 0:
                    p = p[1:len(p) - 1]
                    mvals = [x[1:len(x) - 1] for x in mvals]
                    mvals_sd = [x[1:len(x) - 1] for x in mvals_sd]
                d = self.node_depth[i]
                for j in range(len(self.metric_names)):
                    if d in depths_taken[j]:
                        lab = None
                    else:
                        lab = f'Depth {d}'
                    depths_taken[j].add(d)
                    axs[j].scatter(p, mvals[j], label=lab, color=cmap_depth[d % 10], zorder=10)
                    axs[j].errorbar(p, mvals[j], yerr=mvals_sd[j], fmt='none', ecolor='black', capsize=2, zorder=-10,
                                    alpha=0.3)
            for i in range(len(self.metric_names)):
                name = self.metric_names[i]
                rng = self.metric_ranges[i]
                axs[i].set_title(f'Overall Progress for {name}')
                axs[i].set_ylabel(f'{name}')
                axs[i].set_xlabel(f'{self.parameter_names[0]}')
                x1, x2 = axs[i].get_xlim()
                axs[i].fill_between([x1, x2], [rng[0], rng[0]], [rng[1], rng[1]], color='black', alpha=0.3, zorder=-20)
                axs[i].legend()
                figs[i].savefig(f'{location}/{self.group_id}_{name}_overall_progress.png')
                plt.close(figs[i])
            plt.close('all')

    # To visualize the stochasticity of the runs in relation to the size of the target range
    def plot_sd(self, location):
        for m in self.metric_names:
            temp = []
            temp_sd = []
            for i in self.metric_values_nodes_sd:
                temp_sd += [x for x in i[m].flatten() if not np.isnan(x)]
            for i in self.metric_values_nodes:
                temp += [x for x in i[m].flatten() if not np.isnan(x)]
            cmap = cm.get_cmap('viridis', 256)
            mx, mn = np.nanmax(temp), np.nanmin(temp)
            rng = self.metric_ranges[self.metric_names.index(m)]
            mx = max(mx, max(rng))
            mn = min(mn, min(rng))
            r1, r2 = (rng[0] - mn) / (mx - mn), (rng[1] - mn) / (mx - mn)
            r1, r2 = int(r1 * 256), int(r2 * 256)
            cmapnew = cmap(np.linspace(0, 1, 256))
            pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
            cmapnew[r1:r2 + 1, :] = pink
            cmapnew = ListedColormap(cmapnew)
            norm = Normalize(vmin=mn, vmax=mx)
            fig, ax = plt.subplots()
            temp = sorted(zip(temp, temp_sd))
            ax.scatter(np.arange(len(temp)), np.zeros(len(temp)), c=cmapnew((np.array([x[0] for x in temp]) - mn) / (mx - mn)),
                       s=20, zorder=10)
            ax.errorbar(np.arange(len(temp)), np.zeros(len(temp)), yerr=[x[1] for x in temp], ecolor='black', capsize=3,
                        fmt='none', zorder=0)
            r = abs(rng[1] - rng[0]) / 2
            ax.fill_between(np.arange(len(temp)), [-r for i in range(len(temp))],
                            [r for i in range(len(temp))], alpha=0.2, color=pink, zorder=-1)
            ax.set_xlabel('Sampled Metric Values')
            ax.set_ylabel('Size of +- 1 SD relative to the target range')
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmapnew),
                         orientation='vertical')
            fig.savefig(f'{location}/{self.group_id}_{m}_sd_comparison.png')
            plt.close(fig)

    def plot_all(self, location):
        self.plot_progress(location)
        self.plot_sd(location)

    # Performs the eponymous task :)
    def update_ranges_and_create_children(self, analysis_results, m_1d):
        # analysis results is a list (each index is a run_number) of dicts (each key is a metric, value is [mean, sd])
        i = self.node_run_numbers[self.current_node]
        if (len(analysis_results) <= max(i)) or any([(analysis_results[x] is None) for x in i]):
            return  # because all the necessary analysis results are not ready yet
        # order: Iterates over each point in the nD grid (n >= 1)
        # run_number_matrix: For each point, associates the corresponding run_number
        order = product(np.arange(self.metric_values_nodes[self.current_node][self.metric_names[0]].shape[0]),
                        repeat=len(self.metric_values_nodes[self.current_node][self.metric_names[0]].shape))
        run_number_matrix = np.zeros(self.metric_values_nodes[self.current_node][self.metric_names[0]].shape, dtype=int)
        for i in self.node_run_numbers[self.current_node]:
            o = next(order, None)
            assert o is not None
            for m in self.metric_names:
                ind = tuple([np.array([x]) for x in o])  # Get the array-index
                self.metric_values_nodes[self.current_node][m][ind] = analysis_results[i][m][0]
                self.metric_values_nodes_sd[self.current_node][m][ind] = analysis_results[i][m][1]
                run_number_matrix[ind] = i
        list_of_nd_arrays = [self.metric_values_nodes[self.current_node][m] for m in self.metric_names]
        upper_list = [i[1] for i in self.metric_ranges]
        lower_list = [i[0] for i in self.metric_ranges]
        iterable_ranges, in_range, point_ids = iterate_over_all_axes(list_of_nd_arrays, upper_list, lower_list)
        if any(in_range.flatten()):  # A successfully optimized point is found
            self.state = 1
            optimal_params = in_range.nonzero()
            pd = self.param_nodes[self.current_node]
            for k in self.parameter_names:
                if len(pd[k]) == 1:
                    self.optimized_values[k] = pd[k]
                    continue
                if self.node_parents[self.current_node] != self.current_node:
                    self.optimized_values[k] = pd[k][optimal_params[0]]
                    self.optimized_metric_values = run_number_matrix[optimal_params[0]]
                    continue
                self.optimized_values[k] = pd[k][optimal_params[self.parameter_names.index(k)]]
                self.optimized_metric_values = run_number_matrix[optimal_params]
            self.node_status[self.current_node] = 1
        elif len(iterable_ranges) == 0:  # No children to create
            if self.current_node == 0:
                self.issues.add('No iterable ranges at root node')
            else:
                assert len(self.metric_names) > 1
                self.issues.add('Non overlapping ranges for the different metrics')
            self.node_status[self.current_node] = -1
            cont = self.dfs_continue()
            if not cont:
                self.state = -1
        elif self.node_depth[self.current_node] == self.max_depth:
            self.issues.add(f'Max Depth exceeded at node {self.current_node}')
            self.node_status[self.current_node] = -1
            cont = self.dfs_continue()
            if not cont:
                self.state = -1
        else:
            def foo(x):  # To avoid using lambda
                return x[2]

            iterable_ranges = sorted(iterable_ranges, key=foo, reverse=True)
            for p1, p2, temp in iterable_ranges:
                new_node = dict()
                point_indices = (np.isin(point_ids, [p1, p2])).nonzero()
                for k in range(len(self.parameter_names)):
                    name = self.parameter_names[k]
                    if len(self.param_nodes[self.current_node][name]) == 1:
                        new_node[name] = self.param_nodes[self.current_node][name]
                        continue
                    if self.node_parents[self.current_node] != self.current_node:
                        k = 0
                        assert len(point_indices) == 1
                    if point_indices[k][0] == point_indices[k][1]:
                        new_node[name] = [self.param_nodes[self.current_node][name][point_indices[k][0]]]
                    else:
                        val1 = self.param_nodes[self.current_node][name][point_indices[k][0]]
                        val2 = self.param_nodes[self.current_node][name][point_indices[k][1]]
                        new_node[name] = np.linspace(val1, val2, m_1d + 2)
                self.param_nodes.append(new_node)
                self.node_children[self.current_node].append(len(self.param_nodes) - 1)
                self.node_children.append([])
                self.node_parents.append(self.current_node)
                self.node_status.append(0)
                self.node_depth.append(self.node_depth[self.current_node] + 1)
                run_numbers_recycled = run_number_matrix[point_indices]  # the run numbers inherited from the parent
                runs = [run_numbers_recycled[0]]
                runs += [np.NaN for i in range(m_1d)]
                runs += [run_numbers_recycled[1]]
                self.node_run_numbers.append(np.array(runs))
                m_nd_vals = dict()
                for j in self.metric_names:
                    m_nd_vals[j] = np.zeros(m_1d + 2, dtype=np.float32)
                    m_nd_vals[j][:] = np.NaN
                self.metric_values_nodes.append(m_nd_vals)
                self.metric_values_nodes_sd.append(copy.deepcopy(m_nd_vals))
            self.node_status[self.current_node] = -1
            cont = self.dfs_continue()
            assert cont


class Optimizer:

    def __init__(self, logger_queue, executor_queue, executor_queue_in, **params):
        # load non-user attributes
        self.initialize_time = time.time()
        self.all_params = params
        self.logger_queue = logger_queue
        self.executor_queue = executor_queue
        self.executor_queue_in = executor_queue_in
        self.state = 0  # 0: running, -1: failed, 1: succeeded
        self.blocks_loaded = 0
        self.commands_loaded = 0
        self.current_blocks = []
        self.currently_running = 0
        self.analysis_results = []
        self.progress_tqdms = []
        self.progress_printing_status = 0
        self.run_number_job_id_map = []
        # load compulsory user-dependent attributes
        self.parameter_names = params['parameters'][0]
        self.parameter_groups = params['parameters'][1]
        self.parameter_ranges = params['parameters'][2]
        self.metric_names = params['metrics'][0]
        self.metric_groups = params['metrics'][1]
        self.metric_ranges = params['metrics'][2]
        self.metric_search_strings = params['metrics'][3]
        # load optional user-dependent attributes
        self.n_frames_per_run = None
        if params['n_frames_per_run'] is not None:
            self.n_frames_per_run = int(params['n_frames_per_run'])
        self.output_path = params['path']
        self.analysis_wrapper = params['analysis_wrapper']
        self.max_np = int(params['max_np'])
        self.n_per_command = int(params['n_per_command'])
        self.command = shlex.split(params['COMMAND'])
        reg = re.search('mpirun.*-np ([0-9]+).*', params['COMMAND'], re.DOTALL)
        if reg:
            if int(reg.group(1)) != self.n_per_command:
                message = (time.time(), 'WARNING',
                           f'n_per_command not same as mpirun np in command {self.n_per_command} X {int(reg.group(1))}',
                           'OPTIMIZER')
                self.logger_queue.put(message)
                warnings.warn('Mpirun detected in command with a different np than the given value for n_per_command',
                              UserWarning)
        if (self.max_np * self.n_per_command) > os.cpu_count():
            message = (time.time(), 'WARNING',
                       f'max_np may exceed CPU count ({os.cpu_count()}) max_np = {self.n_per_command} X {self.max_np}',
                       'OPTIMIZER')
            self.logger_queue.put(message)
            warnings.warn('The given values for max_np/n_per_command may result in more processes than CPU count',
                          ResourceWarning)
        self.m_nd = params['m_nd']
        self.m_1d = int(params['m_1d'])
        self.repeat = int(params['repeat'])
        self.stop_param = params['stopping_param']
        self.stop_eq = params['stopping_eq']
        self.stop_err = params['stopping_err']
        self.verbosity = int(params['verbosity'])
        self.cleanup = int(params['cleanup'])
        self.plotting = int(params['plotting'])
        # pre-setup
        self.num_groups = max(self.parameter_groups)
        self.parameter_group_objs = ParameterGroup.create_groups(self.parameter_names, self.parameter_groups,
                                                                 self.parameter_ranges, self.metric_names,
                                                                 self.metric_groups, self.metric_ranges,
                                                                 self.metric_search_strings, params['max_depth'])
        # announce the birth of the object
        message = (time.time(), 'STATUS', 'Initialized', 'OPTIMIZER')
        self.logger_queue.put(message)
        message = (time.time(), 'DETAILS', f'Total parameter groups = {self.num_groups}', 'OPTIMIZER')
        self.logger_queue.put(message)

    def initial_setup(self):
        total_runs_needed = []
        total_params = 0
        total_metrics = 0
        for i in range(len(self.parameter_group_objs)):
            n_params, n_metrics, n_runs = self.parameter_group_objs[i].initial_setup(self.m_nd, self.m_1d)
            total_params += n_params
            total_metrics += n_metrics
            total_runs_needed += [n_runs]
            message = (
                time.time(), 'DETAILS', f'Parameter Group {i}= {n_params} params {n_metrics} metrics', 'OPTIMIZER')
            self.logger_queue.put(message)
            self.current_blocks.append(None)
        total_runs_needed = max(total_runs_needed)
        m = f'Total= Parameters: {total_params}, Metrics: {total_metrics}, Runs: {total_runs_needed}'
        self.progress_update_messages(m, 2)
        message = (time.time(), 'INFO', 'Initial setup complete', 'OPTIMIZER')
        self.logger_queue.put(message)

    def progress_update_messages(self, message, priority):
        if priority <= self.verbosity:
            print(message)

    def progress_for_frames(self, run_number, frames):
        if self.verbosity < 2:
            return
        if self.progress_printing_status == 0:
            self.progress_tqdms = []
            self.progress_tqdms.append(tqdm(desc='Number of runs finished', total=run_number, unit='run',
                                            bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}'))
            if self.n_frames_per_run is not None:
                self.progress_tqdms.append(tqdm(desc='Average frames across active runs',
                                                total=self.n_frames_per_run, unit='frame'))
                self.progress_tqdms.append(tqdm(desc='Minimum frames across active runs',
                                                total=self.n_frames_per_run, unit='frame'))
                self.progress_tqdms.append(tqdm(desc='Maximum frames across active runs',
                                                total=self.n_frames_per_run, unit='frame',
                                                bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}'))
                temp = [0 for i in range(run_number)]
                self.progress_tqdms.append(temp)
                self.progress_tqdms.append([0, 0, 0])
            else:
                self.progress_tqdms.append([0, run_number])
            self.progress_printing_status = 1
        elif self.progress_printing_status == 1:
            if isinstance(frames, int) and (not (self.n_frames_per_run is None)):
                self.progress_tqdms[-2][run_number] = frames
                pending = [x for x in self.progress_tqdms[-2] if x < self.n_frames_per_run]
                if len(pending) > 0:
                    temp = np.mean(pending)
                    temp2 = np.min(pending)
                    temp3 = np.max(pending)
                    self.progress_tqdms[1].update(temp - self.progress_tqdms[-1][0])
                    self.progress_tqdms[2].update(temp2 - self.progress_tqdms[-1][1])
                    self.progress_tqdms[3].update(temp3 - self.progress_tqdms[-1][2])
                    self.progress_tqdms[-1] = [temp, temp2, temp3]
            elif frames == 'end':
                self.progress_tqdms[0].update(1)
                if self.n_frames_per_run is not None:
                    self.progress_tqdms[-2][run_number] = self.n_frames_per_run
                    pending = [x for x in self.progress_tqdms[-2] if x < self.n_frames_per_run]
                    if len(pending) == 0:
                        self.progress_printing_status = 0
                        self.progress_tqdms[0].close()
                        self.progress_tqdms[1].close()
                        self.progress_tqdms[2].close()
                        self.progress_tqdms[3].close()
                        self.progress_printing_status = 0
                        self.progress_tqdms = []
                    else:
                        temp = np.mean(pending)
                        temp2 = np.min(pending)
                        temp3 = np.max(pending)
                        self.progress_tqdms[1].update(temp - self.progress_tqdms[-1][0])
                        self.progress_tqdms[2].update(temp2 - self.progress_tqdms[-1][1])
                        self.progress_tqdms[3].update(temp3 - self.progress_tqdms[-1][2])
                        self.progress_tqdms[-1] = [temp, temp2, temp3]
                else:
                    self.progress_tqdms[1][0] += 1
                    if self.progress_tqdms[1][0] == self.progress_tqdms[1][1]:
                        self.progress_tqdms[0].close()
                        self.progress_tqdms = []
                        self.progress_printing_status = 0

    def load_block(self):
        for i in range(len(self.current_blocks)):
            if self.current_blocks[i] is None:
                if self.parameter_group_objs[i].state == 0:
                    self.current_blocks[i] = self.parameter_group_objs[i].get_next_block()
                    n = self.parameter_group_objs[i].group_id
                    message = (time.time(), 'DETAILS', f'Loading a new set of runs from param-group {n}', 'OPTIMIZER')
                    self.logger_queue.put(message)
        next_block_size = [len(i[list(i.keys())[0]]) for i in self.current_blocks if i is not None]
        next_block_size = min([x for x in next_block_size])
        next_block = dict()
        for i in range(len(self.current_blocks)):
            if self.current_blocks[i] is not None:
                num_runs = 0
                for k in self.current_blocks[i]:
                    next_block[k] = self.current_blocks[i][k][:next_block_size]
                    num_runs = len(self.current_blocks[i][k])
                if num_runs == next_block_size:
                    self.current_blocks[i] = None
                else:
                    for k in self.current_blocks[i]:
                        self.current_blocks[i][k] = self.current_blocks[i][k][next_block_size:]
            else:
                if self.parameter_group_objs[i].state == 1:
                    n = self.parameter_group_objs[i].group_id
                    message = (time.time(), 'DETAILS', f'Using optimized values for param-group{n}', 'OPTIMIZER')
                    self.logger_queue.put(message)
                    for k in self.parameter_group_objs[i].parameter_names:
                        next_block[k] = np.zeros(next_block_size)
                        next_block[k][:] = self.parameter_group_objs[i].optimized_values[k][0]
                else:
                    n = self.parameter_group_objs[i].group_id
                    message = (time.time(), 'DETAILS', f'Using last-used values for param-group{n}', 'OPTIMIZER')
                    self.logger_queue.put(message)
                    for k in self.parameter_group_objs[i].parameter_names:
                        next_block[k] = np.zeros(next_block_size)
                        next_block[k][:] = self.parameter_group_objs[i].last_used_values[k][1]
        self.run_number_job_id_map = []
        for i in range(next_block_size):
            s = []
            for k in self.parameter_names:
                s += [str(next_block[k][i])]
            c = self.command + s
            for j in range(self.repeat):
                x = self.commands_loaded // self.repeat
                if not os.path.isdir(f'{self.output_path}/output_{x}_{j}'):
                    os.mkdir(f'{self.output_path}/output_{x}_{j}')
                else:
                    message = (time.time(), 'WARNING', f'Folder {self.output_path}/output_{x}_{j} already exists',
                               'OPTIMIZER')
                    self.logger_queue.put(message)
                    warnings.warn(f'Folder {self.output_path}/output_{x}_{j} already exists', RuntimeWarning)
                jb = Job(c + [f'{self.output_path}/output_{x}_{j}'], f'{x}_{j}')
                self.run_number_job_id_map.append(f'{x}_{j}')
                self.commands_loaded += 1
                self.currently_running += 1
                self.executor_queue.put(jb)
        self.progress_for_frames(self.currently_running, None)
        message = (time.time(), 'INFO', f'A new block loaded of size {next_block_size}', 'OPTIMIZER')
        self.logger_queue.put(message)
        self.blocks_loaded += 1

    def handle_processes(self):
        message = (time.time(), 'STATUS', 'Awaiting progress and results', 'OPTIMIZER')
        self.logger_queue.put(message)
        fresh_for_analysis = []
        terminate = False
        while self.currently_running > 0:
            message = self.executor_queue_in.get()
            if isinstance(message, tuple):
                job_id, frames = message
                self.progress_for_frames(self.run_number_job_id_map.index(job_id), int(frames))
                message = (time.time(), 'INFO', f'Received progress report on {job_id}', 'OPTIMIZER')
                self.logger_queue.put(message)
            else:
                self.currently_running -= 1
                self.progress_for_frames(self.run_number_job_id_map.index(message.identifier), 'end')
                fresh_for_analysis.append(f'{self.output_path}/output_{message.identifier}')
                if (message.return_code != 0) and not self.stop_err:
                    warnings.warn('Non-zero return-code received but stopping-on-error is set to False', RuntimeWarning)
                ms = (time.time(), 'INFO', f'Finished {message.identifier} with exit code {message.return_code}',
                      'OPTIMIZER')
                self.logger_queue.put(ms)
                if (message.return_code != 0) and self.stop_err:
                    terminate = True
                    message = (time.time(), 'ERROR', 'Terminating the run due to non-zero exit code', 'OPTIMIZER')
                    self.logger_queue.put(message)
                    print(f'ERROR: Terminating due to a non-zero return-code for one of the runs: {message.identifier}')
        if not terminate:
            terminate = self.analyze(fresh_for_analysis)
        self.clean(fresh_for_analysis)
        message = (time.time(), 'INFO', 'Finished handling the block', 'OPTIMIZER')
        self.logger_queue.put(message)
        return terminate

    def plots(self):
        location = f'{self.output_path}/logs'
        if self.plotting == 0:
            return
        message = (time.time(), 'INFO', 'Plotting', 'OPTIMIZER')
        self.logger_queue.put(message)
        for i in self.parameter_group_objs:
            i.plot_all(location)

    def clean(self, data_to_clean):
        if self.cleanup >= 1:
            message = (time.time(), 'INFO', 'Cleaning stuff', 'OPTIMIZER')
            self.logger_queue.put(message)
            for i in data_to_clean:
                j = i.split('/')[-1].split('output_')[-1]
                rmtree(i)
                if self.cleanup >= 2:
                    os.remove(f'{self.output_path}/logs/temp_file_process_stdout_{j}.txt')
                    os.remove(f'{self.output_path}/logs/temp_file_process_stderr_{j}.txt')

    def analyze(self, jobs_to_analyze):
        message = (time.time(), 'INFO', f'Starting to analyze {len(jobs_to_analyze)} results', 'OPTIMIZER')
        self.logger_queue.put(message)
        self.progress_update_messages(f'Analyzing {len(jobs_to_analyze)} results', 2)
        terminate = False
        jobs_to_analyze = sorted(jobs_to_analyze, key=lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1])))
        progress_bar = None
        warning_jobs = []
        warning_jobs_err = []
        if self.verbosity >= 2:
            progress_bar = tqdm(total=len(jobs_to_analyze), unit='run')
        for i in range(len(jobs_to_analyze) // self.repeat):
            jobs = jobs_to_analyze[self.repeat * i:self.repeat * (i + 1)]
            plot = ''
            if self.plotting > 1:
                plot = f'{self.output_path}/logs'
            values = self.analysis_wrapper(jobs, self.metric_names, self.metric_search_strings, plot)
            error_check, equilibriation_dict, values = values
            unequilibriated = [x for x in equilibriation_dict if not equilibriation_dict[x]]
            equilibriation_check = (len(unequilibriated) == 0)
            if self.verbosity >= 2:
                progress_bar.update(self.repeat)
            if self.stop_eq and (not equilibriation_check):
                message = (time.time(), 'ERROR', f'Failed equilibriation in {unequilibriated}. Terminating', 'OPTIMIZER')
                self.logger_queue.put(message)
                print(f'ERROR: Terminating due to failed equilibriation: {unequilibriated}')
                terminate = True
                warning_jobs += unequilibriated
            elif not equilibriation_check:
                message = (time.time(), 'WARNING', f'Failed equilibriation in {unequilibriated}', 'OPTIMIZER')
                self.logger_queue.put(message)
                warning_jobs += unequilibriated
            if self.stop_err and (not error_check):
                message = (time.time(), 'ERROR', f'Error in analysis {values}. Terminating', 'OPTIMIZER')
                self.logger_queue.put(message)
                print(f'ERROR: Terminating due to error in analysis: {values}')
                terminate = True
                warning_jobs_err += jobs
            elif not error_check:
                message = (time.time(), 'WARNING', f'Error in analysis {values}.', 'OPTIMIZER')
                warnings.warn(f'Error in analysis: {values}', RuntimeWarning)
                self.logger_queue.put(message)
                warning_jobs_err += jobs
            if terminate:
                break
            self.analysis_results.append(values)
        if self.verbosity >= 2:
            progress_bar.close()
        if len(warning_jobs) > 0:
            warnings.warn(f'Failed to equilibriate in the following runs: {warning_jobs}', RuntimeWarning)
        if len(warning_jobs_err) > 0:
            warnings.warn(f'Error in analysis for the following runs: {warning_jobs_err}.', RuntimeWarning)
        message = (time.time(), 'INFO', 'Finished analysis', 'OPTIMIZER')
        self.logger_queue.put(message)
        return terminate

    def update_state(self):
        message = (time.time(), 'INFO', 'Updating for parameter-group states and DFS trees', 'OPTIMIZER')
        self.logger_queue.put(message)
        run_not_finished = False
        for i in self.parameter_group_objs:
            if i.state == 0:
                i.update_ranges_and_create_children(self.analysis_results, self.m_1d)
            self.progress_update_messages(f'Parameter group {i.group_id} has state {i.state}', 1)
            run_not_finished = run_not_finished or (i.state == 0)
        return run_not_finished

    def report(self):
        message = (time.time(), 'INFO', 'Generating report', 'OPTIMIZER')
        self.logger_queue.put(message)
        with open(f'{self.output_path}/logs/report.txt', 'w') as f:
            f.write(f'Total Time Taken: {int(time.time() - self.initialize_time)} seconds\n')
            f.write(f'Total Blocks/Commands loaded: {self.blocks_loaded}/{self.commands_loaded}\n')
            f.write(f'Total Params/Param-groups: {len(self.parameter_names)}/{len(self.parameter_group_objs)}\n')
            f.write(f'Total Metrics: {len(self.metric_names)}\n')
            for i in self.parameter_group_objs:
                f.write(f'Parameter Group ID: {i.group_id}\n')
                f.write(f'\tNumber of parameters: {len(i.parameter_names)}\n')
                f.write(f'\tNumber of metrics: {len(i.metric_names)}\n')
                f.write(f'\tNumber of Nodes run: {len([x for x in i.node_status if i != 0])}\n')
                f.write(f'\tMaximum Depth: {max(i.node_depth)}\n')
                message = ['Group runs not finished', 'Successful', 'Failed'][i.state]
                f.write(f'\tOptimization Status: {message}\n')
                if i.state == 1:
                    temp = [i.node_depth[j] for j in range(len(i.node_depth)) if i.node_status[j] == 1][0]
                    f.write(f'\t\tSuccessful Node Depth: {temp}\n')
                    f.write('\t\tOptimal Parameters with corresponding metric values (and sd):\n')
                    temp = set([len(i.optimized_values[x]) for x in i.optimized_values])
                    assert (len(temp) == 1) or ((len(temp) == 2) and (1 in temp))
                    for p in i.parameter_names:
                        f.write(f'\t\t\tParam {p}: {i.optimized_values[p]}\n')
                    for m in i.metric_names:
                        temp = []
                        for r in i.optimized_metric_values:
                            temp.append(f'{self.analysis_results[r][m][0]} (+- {self.analysis_results[r][m][1]})')
                        temp = ', '.join(temp)
                        f.write(f'\t\t\tMetric {m}: {temp}\n')
                if i.state == -1:
                    pass
                if len(i.issues) > 0:
                    f.write('\tIssues Noted:\n')
                    for num, issue in enumerate(i.issues):
                        f.write(f'\t\t{num}. {issue}\n')
            f.write('_________________\n\n')
            f.write('Input options: \n')
            for p in self.all_params:
                f.write(f'\t{p} : {self.all_params[p]}\n')
