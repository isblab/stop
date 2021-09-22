import matplotlib.pyplot as plt
import numpy as np
import os

def foo(x):
    for i in x:
        if x[i] == 'Total_Score':
            return i

folders = [x for x in os.listdir('./') if os.path.isdir(x)]
for folder in folders:
    subfolders = [x for x in os.listdir(f'{folder}/') if x[:7] == 'output_']
    overall = []
    for subfolder in subfolders:
        path = f'{folder}/{subfolder}/stat.0.out'
        with open(path) as f:
            rd = f.read().split('\n')
            rd = [eval(x) for x in rd if len(x) > 0]
            ind = foo(rd[0])
            rd = [x[ind] for x in rd[1:]]
        overall += rd
    with open(f'{folder}_total_scores_all_models.txt', 'w') as f:
        f.write('\n'.join(overall))

