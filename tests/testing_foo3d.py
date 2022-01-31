import sys
import numpy as np

print(f'Doing {sys.argv[-1]}')
temp = np.array(list(map(float, sys.argv[1:len(sys.argv) - 1])), dtype=float)
a, b, c = temp


def fooa(x, y):
    return (x + y) / 2


def foob(x, y):
    return (x ** 2 + y ** 2) / 4


def fooc(x, y):
    return (x + 0.5 * x ** 2 - y ** 2 + 2 * x * y) / 4


ma = fooa(a - c, b - c)
mb = foob(a - c, b - c)
mc = fooc(a - c, b - c)
with open(f'{sys.argv[-1]}/output.txt', 'w') as f:
    f.write(f'{ma},{mb},{mc}')
print('finished')
