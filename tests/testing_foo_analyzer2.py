import numpy as np


def test_analysis(names_of_files, metric_names, param_search_names, plot):
    assert len(names_of_files) == 1
    with open(f'{names_of_files[0]}/output.txt', 'r') as f:
        rd = f.read().split(',')
        rd = list(map(float, rd))
    t = np.array(rd)

    d = dict()
    metric_names = sorted(metric_names, key=lambda x: int(x.split('m')[-1]))
    for i in range(len(metric_names)):
        d[metric_names[i]] = t[i], 0
    return True, dict(), d
