import matplotlib.pyplot as plt
import numpy as np

gtusc_vals = dict()
for i in range(3):
    label_main = ['actin', 'gtusc_10x', 'gtusc_100x'][i]
    path = ['../imp_systems/actin_tropomyosin/modeling/timing/timing_data.txt',
            '../imp_systems/gtusc_spc110_10x/scripts/sample/timing/timing_data.txt',
            '../imp_systems/gtusc_spc110_100x/scripts/sample/timing/timing_data.txt'][i]
    nframes = [10000, 8000, 8000][i]
    time_gap = [60, 5, 5][i]
    print(f'Initiating {label_main}: ')
    with open(path) as f:
        rd = f.read()
    rd = rd.split('\n')
    rd = [list(map(float, x.split(':'))) for x in rd if len(x) > 0]
    rdn = np.array(rd)
    rdn[:, 2] -= rdn[:, 2].min()
    individual_runs = [rdn[rdn[:, 0] == float(i), :] for i in range(8)]
    high = np.vstack([x[:, 1] for x in individual_runs[:3]])
    low = np.vstack([x[:, 1] for x in individual_runs[3:6]])
    optimal = np.vstack([x[:, 1] for x in individual_runs[6:9]])

    cutoff_time = np.mean(optimal, axis=0).tolist().index(nframes)
    print(f'Cutoff_index = {cutoff_time}\nApproximate time = {individual_runs[0][:, 2][cutoff_time]:^.2f} seconds')
    print(f'Value for low run: {np.mean(low, axis=0)[cutoff_time]}')
    print(f'Value for high run: {np.mean(high, axis=0)[cutoff_time]}')
    if 'gtusc' in label_main:
        gtusc_vals[label_main] = [optimal, high, low]
        continue
    plt.figure()
    cols = {'low': 'red', 'optimal': 'blue', 'high': 'green'}
    for data, label in zip([low, optimal, high], ['low', 'optimal', 'high']):
        datamean = np.mean(data, axis=0)[:cutoff_time + time_gap]
        datastd = np.std(data, axis=0, ddof=1)[:cutoff_time + time_gap]
        datarange = datamean - datastd, datamean + datastd
        plt.plot(np.arange(len(datamean)), datamean, color=cols[label], label=label)
        plt.fill_between(np.arange(len(datamean)), datarange[0], datarange[1], color=cols[label], alpha=0.2)
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.ylabel('Number of frames')
    plt.savefig(f'../paper_imp_figures/timing_{label_main}.svg')
    plt.close()

plt.figure()
optimal = np.vstack([gtusc_vals['gtusc_10x'][0][:, :91], gtusc_vals['gtusc_100x'][0][:, :91]])
low_med = gtusc_vals['gtusc_10x'][2]
high_med = gtusc_vals['gtusc_10x'][1]
low = gtusc_vals['gtusc_100x'][2]
high = gtusc_vals['gtusc_100x'][1]
nframes = 8000
cutoff_overall = np.mean(optimal, axis=0).tolist().index(nframes)
cols = {'med-low': 'red', 'optimal': 'blue', 'med-high': 'green', 'high': 'orange', 'low': 'olive'}
for data, label in zip([low_med, optimal, high_med, high, low], ['med-low', 'optimal', 'med-high', 'high', 'low']):
    datamean = np.mean(data, axis=0)[:cutoff_time + time_gap]
    datastd = np.std(data, axis=0, ddof=1)[:cutoff_time + time_gap]
    datarange = datamean - datastd, datamean + datastd
    plt.plot(np.arange(len(datamean)), datamean, color=cols[label], label=label)
    plt.fill_between(np.arange(len(datamean)), datarange[0], datarange[1], color=cols[label], alpha=0.2)
plt.legend()
plt.xlabel('Time (minutes)')
plt.ylabel('Number of frames')
plt.savefig(f'../paper_imp_figures/timing_gtusc_combined.svg')
plt.close()
