import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline

print(f'Doing {sys.argv[-1]}')
np.random.seed(2021)


def normalize(x):
    start = -np.random.random() * 1.4 + 0.4
    end = np.random.random() * 1.4 + 0.6
    return (x - x.min()) / x.ptp() * (end - start) + start


multiple_foos = [[np.random.random((int(x), int(x))) for i in range(20)] for x in np.linspace(5, 25, 5)]
multiple_foos = [[normalize(x) for x in y] for y in multiple_foos]
multiple_foos = [[RectBivariateSpline(np.linspace(-1, 1, len(x)), np.linspace(-1, 1, len(x)), x, kx=3, ky=3) for x in y]
                 for y in multiple_foos]
temp = np.array(list(map(float, sys.argv[1:len(sys.argv) - 1])), dtype=float)

metrics = []
ind = 0
for fooset in multiple_foos:
    for foo in fooset:
        metrics.append(foo(temp[ind], temp[ind + 1]).flatten()[0])
        ind += 2

with open(f'{sys.argv[-1]}/output.txt', 'w') as f:
    f.write(','.join(list(map(str, metrics))))
print('finished')
