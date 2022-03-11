import os
import re
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
matplotlib.use('agg')


# sanity check function
def check_order(main_order):  # To check that not more than 1 frame reports a 1.0 temperature at a time
    temp = []
    for i in main_order:
        temp += i.tolist()
    if not (len(temp) == len(np.unique(temp))):
        return False, 'Repeated 1.0 temp frame indices'
    if not (max(temp) == (len(temp) - 1)):
        return False, 'Incorrect indices: Some indices may be missing'
    return True, temp


def sort_the_replica_exchanges_lowest_temp(main_order):
    m = []  # To flatten the main_order list
    replica_index = []  # Integer indices given to each replica
    for i in range(len(main_order)):
        m += main_order[i].tolist()
        replica_index += ([i for x in range(len(main_order[i]))])
    sorted_replicas = sorted(zip(m, replica_index))  # Sort the indices according the main_order
    sorted_replicas = [x[1] for x in sorted_replicas]  # extract the indices
    x = [False] + (np.diff(sorted_replicas) != 0).tolist()  # Boolean array marking exchanges
    return sorted_replicas, x


def sort_the_replica_exchanges_all_temp(main_array_replica, inverted_dict_replica):
    x = np.array([0 for i in range(len(main_array_replica[0]))], dtype=np.int32)  # Total exchanges at each step
    for i in main_array_replica:
        # Get the temperatures for each frame for this particular replica
        curr_replica_temp = i[:, inverted_dict_replica['ReplicaExchange_CurrentTemp']].flatten()
        curr_replica_temp = [float(x) for x in curr_replica_temp]
        # Check whenever the temperature changes indicating an exchange
        curr_replica_exchanges = [0] + (np.abs(np.diff(curr_replica_temp)) > 1e-5).tolist()
        curr_replica_exchanges = np.array(curr_replica_exchanges, dtype=np.int32)
        x += curr_replica_exchanges
    return x / 2  # Since each exchange is added twice in the two exchanging replicas


def correct_mc_cumulative(mc_array, min_temp_exchanges):
    # Cumulative number of MonteCarlo steps
    adjusted_mc_array = []
    for sub_mc in mc_array:
        number_of_steps = np.arange(1, len(sub_mc) + 1)
        corrected_array = [sub_mc[0]]
        # contains the "actual" number of accepted steps per inter-frame interval
        nsteps = sub_mc[1:] * number_of_steps[1:] - sub_mc[:len(sub_mc) - 1] * number_of_steps[:len(sub_mc) - 1]
        corrected_array = corrected_array + nsteps.tolist()
        corrected_array = np.array(corrected_array)
        # At all exchanges, the replica changes, and hence, the numbers are not valid
        # TODO: Does the exchange happen before or after the MC steps? This changes which all entries to NaN out
        corrected_array[np.array(min_temp_exchanges)] = np.NAN
        if np.sum(np.isnan(corrected_array)) > 0.8 * len(corrected_array):
            adjusted_mc_array.append(np.array(sub_mc))
            continue
        adjusted_mc_array.append(corrected_array)
    return adjusted_mc_array


def parser(path):  # To parse all the stat files
    inf = np.inf  # to handle parsing infinite scores. This should ideally be fixed in the modeling script
    files = os.listdir(path)
    # Load all the appropriately named files from the folder
    stat_files = [x for x in files if re.search(r'stat[.][0-9]+[.]out', x)]
    stat_replica_files = [x for x in files if re.search(r'stat_replica[.][0-9]+[.]out', x)]
    # Check if there is no missing number in the stat files
    set_a = set([int(i.split('.')[1]) for i in stat_files])
    set_b = set(list(range(len(stat_files))))
    if not (set_a == set_b):
        return False, 'Improper stat file numbering'
    for i in stat_files:  # Check if a correspondingly named replica file is present for every stat file
        x = ['stat_replica'] + i.split('.')[1:]
        if not ('.'.join(x) in stat_replica_files):
            return False, 'Corresponding stat_replica file missing'
    # Enough to assert the reverse of the above
    if not (len(stat_files) == len(stat_replica_files)):
        return False, 'Corresponding stat file missing'
    # Re-creating the array to ensure that the two lists are ordered in the same order
    stat_files = ['stat.' + str(i) + '.out' for i in range(len(stat_files))]
    stat_replica_files = ['stat_replica.' + str(i) + '.out' for i in range(len(stat_files))]
    with open(path + '/' + stat_files[0]) as f:
        rd = f.readline().strip()  # The first line contains headers and environment information
        header_info = r'^.+\'STAT2HEADER_IMP_VERSIONS\'[:][ ]*([\"\'])[{].*[}]\1'
        match = re.match(header_info, rd, flags=re.DOTALL)  # Compare the expected header structure
        if not match:
            return False, "Header structure different"
        header_dict = eval(rd)  # Contains the number -> heading mapping as well as the environ details

    with open(path + '/' + stat_replica_files[0]) as f:
        rd = f.readline().strip()
        header_info = r'^.+\'STAT2HEADER_IMP_VERSIONS\'[:][ ]*([\"\'])[{].*[}]\1'
        match = re.match(header_info, rd, flags=re.DOTALL)
        if not match:
            return False, "Header structure different"
        header_dict_replica = eval(rd)  # Contains the number -> heading mapping as well as the environ details

    # build a correctly ordered trajectory
    main_array_replica = []  # contains the parsed dictionaries according to the stat_replica_files
    for i in stat_replica_files:
        with open(path + '/' + i) as f:
            sub_array = []
            for line in f:
                sub_array.append(eval(line))
                # each element is a dictionary with integer keys
            del sub_array[0]  # discard line #1
            main_array_replica.append(np.vstack([[x[y] for y in range(len(x))] for x in sub_array]))

    inverted_dict = dict()  # For heading -> number mapping
    for i in header_dict:
        if isinstance(i, int):
            inverted_dict[header_dict[i]] = i

    inverted_dict_replica = dict()  # For heading -> number mapping
    for i in header_dict_replica:
        if isinstance(i, int):
            inverted_dict_replica[header_dict_replica[i]] = i

    main_order = []  # Contains the correct order of the different frames in different replica files
    for i in main_array_replica:
        temp = i[:, inverted_dict_replica['ReplicaExchange_CurrentTemp']].flatten() == '1.0'
        main_order.append(np.where(temp)[0])

    success, collated_order = check_order(main_order)  # Flattened list version of main_order
    if not success:
        return False, collated_order
    main_array = []  # contains the parsed dictionaries according to the stat_files
    for i in stat_files:
        with open(path + '/' + i) as f:
            sub_array = []
            for line in f:
                sub_array.append(eval(line))
            del sub_array[0]  # discard line 1
            if len(sub_array) == 0:
                continue
            main_array.append(np.vstack([[x[y] for y in range(len(x))] for x in sub_array]))
    if not (len(set([len(x) for x in main_array_replica])) == 1):
        return False, 'Replica stat files have different number of frames'
    if not (len(collated_order) == sum([len(x) for x in main_array])):
        return False, 'All frames not included in main_order/array'
    if not (len(collated_order) == len(main_array_replica[0])):
        return False, 'All frames not included in main_order/array_replica'
    x = []
    for i in range(len(main_array)):
        x += main_array[i].tolist()
    z = sorted(zip(collated_order, x))
    z = [i[1] for i in z]  # properly ordered list of dictionaries (with stat file fields)
    main_order_2 = np.diff(np.array([int(i[inverted_dict['MonteCarlo_Nframe']]) for i in z]))
    # confirm that MonteCarlo_Nframe matches the order based on temperature 1
    if not ((len(np.unique(main_order_2)) == 1) and (np.unique(main_order_2)[0] == 1)):
        return False, 'Temperature based frame ordering does not match MonteCarlo_Nframe'
    x = []
    for i in range(len(main_array)):
        x += main_array_replica[i][main_order[i]].tolist()
    z_replica = sorted(zip(collated_order, x))
    z_replica = [i[1] for i in z_replica]  # properly ordered list of dictionaries (for temp 1 replica file fields)
    for key in inverted_dict:
        values = [np.isinf(i[inverted_dict[key]]) for i in z]
        if any(values[int(0.1 * len(values)):]):  # arbitrary 10-percent cutoff
            return False, 'Infinity encountered after the first 10-percent of the frames'
        if any(values[:int(0.1 * len(values))]):
            print(f'WARNING: Infinity encountered in {key} in the first 10-percent frames at {path}')
    return True, (z, z_replica, main_array, main_array_replica, inverted_dict, inverted_dict_replica, main_order)


def check_equilibriation(series, plot, name, sigma=2, piece=0.25):
    piece_length = int(piece * len(series))
    final_part = series[-piece_length:]
    second_final = series[-2 * piece_length:-piece_length]
    mn, sd = np.mean(final_part), np.std(final_part)
    mn_2, sd_2 = np.mean(second_final), np.std(second_final)
    if np.abs(mn_2 - mn) < (sigma * min(sd, sd_2)):
        equil = True
    else:
        equil = False
    if plot:
        if not os.path.isdir(f'{plot}/equilibriated_plots'):
            os.mkdir(f'{plot}/equilibriated_plots')
        if not os.path.isdir(f'{plot}/non_equilibriated_plots'):
            os.mkdir(f'{plot}/non_equilibriated_plots')
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(second_final)), second_final, color='black', alpha=0.4, zorder=5)
        ax.plot(np.arange(len(final_part)) + len(second_final), final_part, color='red', alpha=0.4, zorder=5)
        ax.plot([0, 2 * piece_length], [mn, mn], color='red', zorder=10)
        ax.plot([0, 2 * piece_length], [mn - sigma * sd, mn - sigma * sd], color='red', zorder=10, linestyle='--')
        ax.plot([0, 2 * piece_length], [mn + sigma * sd, mn + sigma * sd], color='red', zorder=10, linestyle='--')
        ax.plot([0, 2 * piece_length], [mn_2, mn_2], color='black', zorder=10)
        ax.plot([0, 2 * piece_length], [mn_2 - sigma * sd_2, mn_2 - sigma * sd_2], color='black', zorder=10,
                linestyle='--')
        ax.plot([0, 2 * piece_length], [mn_2 + sigma * sd_2, mn_2 + sigma * sd_2], color='black', zorder=10,
                linestyle='--')
        ax.set_ylabel('Total Score (with equilibriation checks)')
        ax.set_xlabel('Final half frames')
        if equil:
            fig.savefig(f'{plot}/equilibriated_plots/equilibriation_check_last_50p_{name}.png')
        else:
            fig.savefig(f'{plot}/non_equilibriated_plots/equilibriation_check_last_50p_{name}.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(np.arange(int(0.9 * len(series))), series[-int(0.9 * len(series)):], color='black', zorder=5)
        ax.set_ylabel('Total Score')
        ax.set_xlabel('Last 90 percent frames')
        fig.savefig(f'{plot}/equilibriation_check_last_90p_{name}.png')
        plt.close(fig)
    return equil


def DefaultAnalysis(names_of_files, metric_names, param_search_names, plot):
    results = dict()
    temp = [[] for _ in range(len(metric_names))]
    equilibriation = dict()
    for i in names_of_files:
        success, vals = parser(i)
        if not success:
            return False, dict(), vals
        z, z_replica, main_array, main_array_replica, inverted_dict, inverted_dict_replica, main_order = vals
        sorted_replicas, exchange_indices = sort_the_replica_exchanges_lowest_temp(main_order)
        exchange_counts_all = sort_the_replica_exchanges_all_temp(main_array_replica, inverted_dict_replica)
        tot_score = [float(j[inverted_dict['Total_Score']]) for j in z]
        equilibriation[i] = check_equilibriation(tot_score, plot, i.split('/')[-1])
        for ind, search_string in enumerate(param_search_names):
            stat_key = any([re.search(search_string, str(x)) for x in inverted_dict.keys()])
            replica_key = any([re.search(search_string, str(x)) for x in inverted_dict_replica.keys()])
            if search_string == 'MinTempReplicaExchangeRatio':
                temp[ind].append(np.mean(exchange_indices[len(exchange_indices) // 2:]))
            elif search_string == 'AllTempReplicaExchangeRatio':
                temp[ind].append(np.mean(exchange_counts_all[len(exchange_counts_all) // 2:]))
            elif not (stat_key or replica_key):
                return False, equilibriation, f"Key not found: {search_string}"
            if stat_key:
                header_list = [x for x in inverted_dict.keys() if re.search(search_string, str(x))]
                unfiltered_array_list = [[float(j[inverted_dict[i]]) for j in z] for i in header_list]
                adjusted_unfiltered_array_list = correct_mc_cumulative(unfiltered_array_list, exchange_indices)
                last_50 = np.array([x[-len(x) // 2:] for x in adjusted_unfiltered_array_list])
                last_50 = np.mean(last_50, axis=0)  # average across matching keys
                temp[ind].append(np.nanmean(last_50.flatten()))  # average across frames
            elif replica_key:
                header_list = [x for x in inverted_dict_replica.keys() if re.search(search_string, str(x))]
                replica_list = []
                for replica in main_array_replica:
                    unfiltered_array_list = [[float(j[inverted_dict[i]]) for j in replica] for i in header_list]
                    adjusted_unfiltered_array_list = correct_mc_cumulative(unfiltered_array_list, exchange_indices)
                    last_50 = np.array([x[-len(x) // 2:] for x in adjusted_unfiltered_array_list])
                    last_50 = np.mean(last_50, axis=0)  # average across matching keys
                    replica_list.append(last_50)
                replica_list = np.mean(np.array(replica_list, dtype=float), axis=0)  # average across replicas
                temp[ind].append(np.nanmean(replica_list.flatten()))  # average across frames
    for name, tempvals in zip(metric_names, temp):
        results[name] = (np.mean(tempvals), np.std(tempvals))  # average across runs
    return True, equilibriation, results
