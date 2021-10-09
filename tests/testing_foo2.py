import sys
import numpy as np

if len(sys.argv) > 1:
    print(f'Doing {sys.argv[-1]}')
    x1 = sys.argv[-1].split('_')[-2]
    x2 = sys.argv[-1].split('_')[-1]
    np.random.seed(2021)
    zdict = np.random.choice([0.3, 0.6], 1000)
    np.random.seed(int(x1) * 10 + int(x2))
    temp = np.array(list(map(float, sys.argv[1:len(sys.argv) - 1])), dtype=float)
    a, b, c = temp / np.arange(1, len(temp) + 1)


    def foo(x):
        return -(x ** 2) + 1 + np.random.normal(0, 0.05)


    def fooma(a):
        return foo(a)


    def foomb(b, c):
        return foo((b + c) / 2)


    ma = fooma(a)
    mb = foomb(b, c)
    with open(f'{sys.argv[-1]}/output.txt', 'w') as f:
        f.write(f'{ma},{mb}')
    print('finished')
