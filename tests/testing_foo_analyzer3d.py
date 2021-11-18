import numpy as np


def test_analysis(names_of_files, metric_names, param_search_names, plot):
    t = None
    for i in names_of_files:
        with open(f'{i}/output.txt', 'r') as f:
            rd = f.read().split(',')
            rd = list(map(float, rd))
        if t is None:
            t = np.array(rd)[np.newaxis, :]
        else:
            t = np.vstack([t, np.array(rd)])
    d = dict()
    metric_names = sorted(metric_names)
    for i in range(len(metric_names)):
        d[metric_names[i]] = np.mean(t, axis=0)[i], np.std(t, axis=0)[i]
    return True, dict(), d
