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
    a, b, c, d, f, h, k = temp / np.arange(1, len(temp) + 1)


    def foo(x):
        return -(x ** 2) + 1 + np.random.normal(0, 0.05)


    def foo2(x):
        return x * 0.5 + 0.7 + np.random.normal(0, 0.05)

    def foo3(x):
        return 1 - x ** 3 - 1.2 * x ** 2 + 0.5 * x + np.random.normal(0, 0.05)


    def fooma(a):
        return foo(a)


    def foomb(b, c):
        return foo((b + c) / 2)


    def foomd(d):
        return foo(d)


    def foome(d):
        return foo2(d)

    def foomf(f):
        return foo(f)

    def foomg(f):
        return foo3(f)

    def foomh(h):
        return foo(h - 0.5)

    def foomk(k):
        z = zdict[np.argmin(np.abs(np.linspace(-1, 1, 1000) - k))]
        return z + (np.random.random() - 0.5) * 0.1

    ma = fooma(a)
    mb = foomb(b, c)
    md = foomd(d)
    me = foome(d)
    mf = foomf(f)
    mg = foomg(f)
    mh = foomh(h)
    mk = foomk(k)
    with open(f'{sys.argv[-1]}/output.txt', 'w') as f:
        f.write(f'{ma},{mb},{md},{me},{mf},{mg},{mh},{mk}')
    print('finished')
