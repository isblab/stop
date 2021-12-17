import queue
import os
import time
import subprocess
import re
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
from itertools import product
from analyzer import DefaultAnalysis


# Multi-processing logger with parallel inputs from multiple processes (via mp-queue)
class MPLogger:

    def __init__(self, filename, logger_queue, death_event, mode='w', max_wait=10):
        self.file = open(filename, mode=mode, buffering=1)
        self.process_queue = logger_queue
        self.max_wait = max_wait
        self.death_event = death_event
        message = (time.time(), 'STATUS', 'Initialized', 'LOGGER')
        self.log(message)

    def log(self, message_tuple):
        t, kind, message, process_type = message_tuple
        timestamp = time.asctime(time.localtime(t))
        self.file.write(f'{timestamp:^25}::{process_type:^10}::{kind:^15}::{message}\n')

    def handle_processes(self):
        message = (time.time(), 'STATUS', 'Starting to listen for logging requests.', 'LOGGER')
        self.log(message)
        while not self.death_event.is_set():
            try:
                message = self.process_queue.get(True, timeout=self.max_wait)
            except queue.Empty:
                message = (time.time(), 'INFO', 'Sitting idle. Nothing to do.', 'LOGGER')
            self.log(message)
        message = (
            time.time(), 'STATUS',
            f'Received death signal. Waiting {self.max_wait * 2} seconds to gather all dying messages.',
            'LOGGER')
        self.log(message)
        time.sleep(self.max_wait)
        while not self.process_queue.empty():
            try:
                message = self.process_queue.get_nowait()
            except queue.Empty:
                message = (time.time(), 'ERROR', 'Logging messaged promised but none received.', 'LOGGER')
            self.log(message)
        message = (time.time(), 'STATUS', 'Dying.', 'LOGGER')
        self.log(message)
        self.file.close()


# Holds the details of a single command execution (IMP commands)
class Job:

    def __init__(self, c, identifier):
        self.subprocess = None
        self.file_reader_stdout = None
        self.file_reader_stderr = None
        self.file_writer_stdout = None
        self.file_writer_stderr = None
        self.stdout = ''
        self.stderr = ''
        self.identifier = identifier
        self.return_code = None
        self.command = c


# Manages the execution of the IMP commands, communicating with the outputs and following on the progress
class Executor:

    def __init__(self, death_event, logger_queue, manager_queue, manager_queue_out, max_p, path, max_wait=10):
        self.logger_queue = logger_queue
        self.path = path
        self.manager_queue = manager_queue
        self.manager_queue_out = manager_queue_out
        self.death_event = death_event
        self.max_p = max_p
        self.currently_active = 0
        self.max_wait = max_wait
        self.last_update = time.time()
        self.alive = []
        # assumes this pattern outputted by the PMI's rex-exchange0 macro
        self.regex = re.compile('--- frame [0-9]+ score')
        message = (time.time(), 'STATUS', 'Initialized', 'EXECUTOR')
        self.logger_queue.put(message)

    def get_temp_file_handle(self, job_obj):
        s1 = f'{self.path}/temp_file_process_stdout_{job_obj.identifier}.txt'
        s2 = f'{self.path}/temp_file_process_stderr_{job_obj.identifier}.txt'
        return s1, s2

    @staticmethod
    def start_new(command, temp_file, temp_file2):
        f = open(temp_file, 'w')
        f2 = open(temp_file2, 'w')
        f_reader = open(temp_file, 'r')
        f_reader2 = open(temp_file2, 'r')
        s = subprocess.Popen(command, bufsize=1, text=True, stdout=f, stderr=f2)
        return s, f_reader, f_reader2, f, f2

    def progress_report(self, job_obj):
        matches = self.regex.findall(job_obj.stdout)
        self.manager_queue_out.put((job_obj.identifier, len(matches)))

    def talk_to_processes(self):
        message = (time.time(), 'INFO', f'Initiating talking to the running processes {len(self.alive)}', 'EXECUTOR')
        self.logger_queue.put(message)
        new_alive = []
        for i in range(len(self.alive)):
            self.alive[i].stdout += self.alive[i].file_reader_stdout.read()
            self.alive[i].stderr += self.alive[i].file_reader_stderr.read()
            self.progress_report(self.alive[i])
            if self.alive[i].subprocess.poll() is None:
                new_alive.append(self.alive[i])
            else:
                self.currently_active -= 1
                self.alive[i].stdout += self.alive[i].file_reader_stdout.read()
                self.alive[i].stderr += self.alive[i].file_reader_stderr.read()
                self.alive[i].file_reader_stdout.close()
                self.alive[i].file_reader_stderr.close()
                self.alive[i].file_writer_stderr.close()
                self.alive[i].file_writer_stdout.close()
                self.alive[i].return_code = self.alive[i].subprocess.returncode
                self.alive[i].subprocess = None
                self.alive[i].file_reader_stdout = None
                self.alive[i].file_reader_stderr = None
                self.alive[i].file_writer_stderr = None
                self.alive[i].file_writer_stdout = None
                self.manager_queue_out.put(self.alive[i])
        message = (time.time(), 'DETAILS', f'Out of {len(self.alive)} processes {len(new_alive)} are still running',
                   'EXECUTOR')
        self.logger_queue.put(message)
        self.alive = new_alive

    def handle_processes(self):
        last_talk = time.time()
        message = (time.time(), 'STATUS', 'Starting to listen for jobs', 'EXECUTOR')
        self.logger_queue.put(message)
        while not self.death_event.is_set():
            if self.currently_active < self.max_p:
                try:
                    job_obj = self.manager_queue.get(True, timeout=self.max_wait)
                    message = (time.time(), 'INFO', 'Received a job to run', 'EXECUTOR')
                    self.logger_queue.put(message)
                    s, f, f_err, fw, fw_err = self.start_new(job_obj.command, *self.get_temp_file_handle(job_obj))
                    self.currently_active += 1
                    job_obj.subprocess = s
                    job_obj.file_reader_stdout = f
                    job_obj.file_reader_stderr = f_err
                    job_obj.file_writer_stdout = fw
                    job_obj.file_writer_stderr = fw_err
                    self.alive.append(job_obj)
                except queue.Empty:
                    message = (time.time(), 'INFO',
                               f'No Commands received. {self.max_p - self.currently_active}/{self.max_p} slots empty',
                               'EXECUTOR')
                    self.logger_queue.put(message)
            else:
                if (time.time() - self.last_update) < self.max_wait:
                    continue
                self.last_update = time.time()
                message = (time.time(), 'INFO', 'Running full. Not looking for more jobs.', 'EXECUTOR')
                self.logger_queue.put(message)
            if (time.time() - last_talk) < self.max_wait:
                continue
            last_talk = time.time()
            self.talk_to_processes()
        message = (time.time(), 'STATUS', 'Received death signal. Dying.', 'EXECUTOR')
        self.logger_queue.put(message)


# To find the "best" iterable range given a line of values based on interpolated values lying in the target range
def find_1d_interpolated_width(a, upper, lower, n=100):
    if len(a) > 3:
        kind = 'cubic'
    elif len(a) >= 2:
        kind = 'quadratic'
    else:
        kind = 'linear'
    foo = interp1d(np.arange(len(a)), a, kind=kind)
    range_dict = dict()
    for i in range(1, len(a)):
        queries = np.linspace(a[i - 1], a[i], n)
        values = foo(queries)
        # number of points lying in the range is the proxy for how good is this range for subsequent iteration
        range_dict[(i - 1, i)] = np.sum(np.logical_and(values <= upper, values >= lower)) / 100
    return range_dict


# Same as above but for multiple metrics
def find_1d_interpolated_overlapping_width(a_list, upper_list, lower_list, n=100):
    if len(a_list[0]) > 3:
        kind = 'cubic'
    elif len(a_list[0]) >= 2:
        kind = 'quadratic'
    else:
        kind = 'linear'
    foos = [interp1d(np.arange(len(a)), a, kind=kind) for a in a_list]
    range_dict = dict()
    for i in range(1, len(a_list[0])):
        queries = [np.linspace(i - 1, i, n) for a in range(len(a_list))]
        values = [foo(queries) for foo in foos]
        within_range_check = [np.logical_and(values[j] <= upper_list[j], values[j] >= lower_list[j]) for j in
                              range(len(values))]
        within_range_check = np.logical_and.reduce(within_range_check)
        range_dict[(i - 1, i)] = np.sum(within_range_check) / 100
    return range_dict


# Find iterable ranges given a single line of metric values
def find_1d_ranges(a, upper_lim, lower_lim, point_ids):
    in_range = np.logical_and(a <= upper_lim, a >= lower_lim)
    greater = a > upper_lim
    lower = a < lower_lim
    range_dict = find_1d_interpolated_width(a, upper_lim, lower_lim)
    iterable_ranges = []
    for i in range(1, len(a)):
        if (greater[i - 1] ^ greater[i]) or (lower[i - 1] ^ lower[i]) or any([in_range[i - 1], in_range[i]]):
            iterable_ranges.append((point_ids[i - 1], point_ids[i], range_dict[(i - 1, i)]))
    return iterable_ranges, in_range


# Same as above but for multiple metrics
def find_overlapping_ranges(a_list, upper_list, lower_list, point_ids):
    in_range_org = [np.logical_and(a_list[j] <= upper_list[j], a_list[j] >= lower_list[j]) for j in range(len(a_list))]
    in_range = np.logical_and.reduce(in_range_org)
    greater = [a_list[i] > upper_list[i] for i in range(len(a_list))]
    lower = [a_list[i] < lower_list[i] for i in range(len(a_list))]
    range_dict = find_1d_interpolated_overlapping_width(a_list, upper_list, lower_list)
    iterable_ranges = []
    for i in range(1, len(a_list[0])):
        temp = [(greater[x][i - 1] ^ greater[x][i]) or (lower[x][i - 1] ^ lower[x][i]) or (
                in_range_org[x][i - 1] or in_range_org[x][i]) for x in range(len(a_list))]
        good_range = np.logical_and.reduce(temp)
        if good_range or any([in_range[i - 1], in_range[i]]):
            iterable_ranges.append((point_ids[i - 1], point_ids[i], range_dict[(i - 1, i)]))
    return iterable_ranges, in_range


# Given any input nd-grid of metric values, find all iterable ranges
def iterate_over_all_axes(list_of_nd_arrays, upper_list, lower_list):
    # Uniquely identify each point on the grid
    point_ids = np.arange(np.prod(list_of_nd_arrays[0].shape)).reshape(list_of_nd_arrays[0].shape)
    in_range_main = np.zeros(point_ids.shape, dtype=bool)
    # Choose all-but-one indices -> freeze their values to get a "line" in the grid
    order = product(np.arange(list_of_nd_arrays[0].shape[0]), repeat=len(list_of_nd_arrays[0].shape) - 1)
    all_iterable_ranges = []
    for i in order:
        # For each permutation of the freezed values of n-1 dimensions, take the remaining "line" dimension
        # as one of the n available dimensions sequentially (this exhaustively covers all the "lines")
        for line_index in range(len(list_of_nd_arrays[0].shape)):
            # Generate the indices to index the "line" from the nD grid
            indexes = [i[:line_index] + (x,) + i[line_index:] for x in range(list_of_nd_arrays[0].shape[0])]
            # Create a set of 1D lists of all the metric values along that line
            a_list = [np.array([x[ii] for ii in indexes]) for x in list_of_nd_arrays]
            # Identify the point ids corresponding to this "line"
            sub_point_ids = np.array([point_ids[ii] for ii in indexes])
            # Find the iterable ranges
            sub_iterable_ranges, sub_in_range = find_overlapping_ranges(a_list, upper_list, lower_list, sub_point_ids)
            for ii in range(len(indexes)):
                # Check if any of the points are in-range
                # TODO: Check for overlapping points match
                in_range_main[indexes[ii]] = sub_in_range[ii]
            all_iterable_ranges += sub_iterable_ranges
    return all_iterable_ranges, in_range_main, point_ids


# The default values for the input options
option_defaults = dict([('m_1d', 3), ('max_np', None), ('verbosity', 2), ('stopping_eq', 0),
                        ('stopping_param', 0), ('cleanup', 3), ('plotting', 3), ('stopping_err', 1),
                        ('max_wait', 10), ('repeat', 3), ('n_frames_per_run', None),
                        ('n_per_command', 1), ('analysis_wrapper', None), ('m_2d', None),
                        ('m_3d', None), ('m_4d', None), ('m_5d', None), ('max_depth', 5), ('path', './')])

# The regex format checkers for the input options
option_checks = dict([('m_1d', ['[0-9]+']), ('max_np', ['[0-9]+']), ('verbosity', ['(0|1|2)']),
                      ('stopping_eq', ['(0|1)']), ('stopping_param', ['(0|1)']), ('cleanup', ['(0|1|2|3)']),
                      ('plotting', ['(0|1|2|3)']), ('max_wait', ['[0-9]+']),
                      ('repeat', ['[0-9]+']), ('n_per_command', ['[0-9]+']),
                      ('n_frames_per_run', ['[0-9]+']),
                      ('analysis_wrapper', [r'^[0-9a-zA-Z_\-.]+$', r'^[0-9a-zA-Z_\-.]+$']),
                      ('METRIC', [r'^[0-9a-zA-Z_\-.]+$', r'^[\-]?[0-9]+[\.]?[0-9]*[ ,]+[\-]?[0-9]+[\.]?[0-9]*$', '.*']),
                      ('PARAM', [r'^[0-9a-zA-Z_\-.]+$', '.*', r'^[\-]?[0-9]+[\.]?[0-9]*[ ,]+[\-]?[0-9]+[\.]?[0-9]*$']),
                      ('COMMAND', ['.*']), ('m_2d', ['[0-9]+']), ('m_3d', ['[0-9]+']), ('m_4d', ['[0-9]+']),
                      ('m_5d', ['[0-9]+']), ('max_depth', ['[0-9]+']),
                      ('path', [r'^[/0-9a-zA-Z_\-.]+$']), ('stopping_err', ['(0|1)'])])


# Creating the Parameter-Metric groups based on Union-find
# Each group is a disjoint connected subgraph of the original bipartite Parameter-Metric graph
def create_sets(parameter_details):
    def find(x):
        if parents[x] == -1:
            return x
        else:
            parents[x] = find(parents[x])
            return parents[x]

    def union(x, y):
        if find(x) != find(y):
            parents[y] = x

    parents = [-1 for i in parameter_details]
    parameter_names = [x[0].strip() for x in parameter_details]
    # Each set contains the metrics affected by the corresponding parameter
    metric_sets = [set([y.strip() for y in x[1].split(',')]) for x in parameter_details]
    # The reverse mapping (i.e. metric -> parameters)
    metric_p = defaultdict(set)
    # Populate the metric_p map
    for i in range(len(metric_sets)):
        for j in metric_sets[i]:
            metric_p[j].add(i)
    # Merge the indices of the parameters which have overlapping metrics in the metric_sets
    for k in metric_p:
        values_to_merge = list(metric_p[k])
        for i in range(1, len(values_to_merge)):
            union(values_to_merge[0], values_to_merge[i])
    final_groups = dict()  # Assign an integer group to each parameter
    for i in range(len(parameter_names)):
        final_groups[parameter_names[i]] = find(i)
    old_values = sorted(list(set(final_groups.values())))
    # Reindex the groups to have integer labels from 1 to n_groups
    rebased_values = [i for i in range(len(old_values))]
    for i in final_groups:
        final_groups[i] = rebased_values[old_values.index(final_groups[i])]
    return final_groups, metric_sets


# Main input option file parsing utility
def parse_file(filename):
    fields = []
    with open(filename, 'r') as f:
        for line in f:
            fields.append(line.strip().split(':'))
    param_mapping = dict()
    compulsory = ['COMMAND', 'PARAM', 'METRIC']  # These do not have a default value
    for i in compulsory:
        if i not in [field[0].strip() for field in fields]:
            print(f'PARSING ERROR: No {i} option set in param file')
            quit(1)
    for field in fields:
        f = field[0].strip()
        if not ((f in compulsory) or (f in option_defaults)):
            print(f'PARSING ERROR: Unexpected token in the param file ({f})')
            quit(1)
        else:
            check = option_checks[f]
            if len(check) != (len(field) - 1):
                print(f'PARSING ERROR: Option {f} requires {len(check)} fields but {len(field) - 1} were given')
                quit(1)
            for i in range(len(check)):  # regex check
                if not re.search(check[i], field[i + 1].strip(), re.DOTALL):
                    print(f'PARSING ERROR: Unexpected format for option {f} field {i + 1}')
                    quit(1)
        if f in compulsory[1:]:  # will be handled separately later
            continue
        if len(field) == 2:
            param_mapping[f] = field[1].strip()
        else:
            param_mapping[f] = [x.strip() for x in field[1:]]
    for i in option_defaults:  # Set the default values for options not overridden in the input file
        if i not in param_mapping:
            param_mapping[i] = option_defaults[i]
    if param_mapping['max_np'] is None:
        param_mapping['max_np'] = os.cpu_count() // param_mapping['n_per_command']
    if param_mapping['analysis_wrapper'] is None:
        param_mapping['analysis_wrapper'] = DefaultAnalysis
    else:
        file, foo = param_mapping['analysis_wrapper']
        str_to_exec = f'from {file} import {foo}'
        param_mapping['analysis_wrapper'] = str_to_exec
    if int(param_mapping['m_1d']) < 2:
        print('PARSING ERROR: option m_1d must be greater than 2')
        quit(1)
    for i, x in enumerate(['m_2d', 'm_3d', 'm_4d', 'm_5d']):
        n = i + 2
        if param_mapping[x] is None:
            param_mapping[x] = max(3, np.floor((3 * int(param_mapping['m_1d'])) ** (1 / n)))
        elif int(param_mapping[x]) < 2:
            print(f'PARSING ERROR: option m_{n}d must be greater than 2')
            quit(1)
    metric_details = [field[1:] for field in fields if field[0].strip() == 'METRIC']
    m_names = [x[0].strip() for x in metric_details]
    m_ranges = [[float(y.strip()) for y in x[1].split(',')] for x in metric_details]
    m_search_strings = [x[2].strip() for x in metric_details]
    parameter_details = [field[1:] for field in fields if field[0].strip() == 'PARAM']
    p_names = [x[0].strip() for x in parameter_details]
    p_ranges = [[float(y.strip()) for y in x[2].split(',')] for x in parameter_details]
    for i in range(len(parameter_details)):
        param_name = parameter_details[i][0].strip()
        affected_metrics = [y.strip() for y in parameter_details[i][1].split(',')]
        # check for validity of the metric names
        for j in affected_metrics:
            if j not in m_names:
                print(f'PARSING ERROR: Unknown metric {j} in parameter {param_name}')
                quit(1)
    parameters_group_mapping, metric_sets = create_sets(parameter_details)
    p_groups = [None for i in p_names]
    m_groups = [None for i in m_names]
    for i in range(len(p_names)):
        nm = p_names[i]
        m_set = metric_sets[i]
        group = parameters_group_mapping[nm]
        p_groups[i] = group
        for j in m_set:
            if not ((m_groups[m_names.index(j)] is None) or (m_groups[m_names.index(j)] == group)):
                print('PARSING INTERNAL ERROR: Group mismatch. Union-Find failure?')
                quit(1)
            m_groups[m_names.index(j)] = group
    param_mapping['parameters'] = [p_names, p_groups, p_ranges]
    param_mapping['metrics'] = [m_names, m_groups, m_ranges, m_search_strings]
    param_mapping['m_nd'] = [int(param_mapping[f'm_{i}d']) for i in range(2, 6)]
    return param_mapping
