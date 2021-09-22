import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

low = pd.read_csv('low/filter/model_ids_scores.txt', sep=' ')
optimal = pd.read_csv('optimal/filter/model_ids_scores.txt', sep=' ')
high = pd.read_csv('high/filter/model_ids_scores.txt', sep=' ')
high_timed = pd.read_csv('high_timed/filter/model_ids_scores.txt', sep=' ')
low_timed = pd.read_csv('low_timed/filter/model_ids_scores.txt', sep=' ')

def foo(x, y, cutoffs=np.linspace(-1,0,20)):
    results = []
    mn = np.mean(y)
    if len(np.unique(y)) == 1:
    	return [x == y[0] for i in cutoffs]
    sd = np.std(y)
    return [x <= (mn + i * sd) for i in cutoffs]
    
def foo_reduce(x):
    final = []
    for i in range(len(x[0])):
        y = [j[i] for j in x]
        y = np.logical_and.reduce(y)
        final.append(np.sum(y))
    return final
    	
        

print(f'Number of Models: optimal={len(optimal)}, high={len(high)}, low={len(low)}, high_timed={len(high_timed)}, low_timed={len(low_timed)}')

cutoff_comparisons = []
cutoff_comparisons_high = []
cutoff_comparisons_high_timed = []
for x in high.columns.to_numpy()[4:]:
    mn = np.mean(optimal[x])
    sd = np.std(optimal[x])
    rng = mn + np.array([-1, 1]) * sd * 3
    a = mn + sd
    b = mn - sd
    print(f'Mean/SD/Cutoff for {x} is {mn:^.2f} / {sd:^.2f} / {mn - 0.25 * sd:^.2f}')
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].hist(high[x], bins=100, alpha=0.2, label='high', range=rng)
    ax[0].hist(optimal[x], bins=100, alpha=0.2, label='optimal', range=rng)
    ax[0].hist(low[x], bins=100, alpha=0.2, label='low', range=rng)
    ax[0].axvline(a, color='black', linestyle='--')
    ax[0].axvline(b, color='black', linestyle='--')
    ax[0].legend()
    
    ax[1].hist(high_timed[x], bins=100, alpha=0.2, label='high_timed', range=rng)
    ax[1].hist(optimal[x], bins=100, alpha=0.2, label='optimal', range=rng)
    ax[1].hist(low_timed[x], bins=100, alpha=0.2, label='low_timed', range=rng)
    ax[1].axvline(a, color='black', linestyle='--')
    ax[1].axvline(b, color='black', linestyle='--')
    ax[1].legend()
    
    fig.suptitle(x)
    plt.savefig(f'comparative_plots/comparative_{x}.png')
    plt.close()
    if ('CrossLinking' not in x) and ('Total_Score' not in x):
        cutoff_comparisons.append(foo(optimal[x].to_numpy(), optimal[x].to_numpy()))
        cutoff_comparisons_high.append(foo(high[x].to_numpy(), optimal[x].to_numpy()))
        cutoff_comparisons_high_timed.append(foo(high_timed[x].to_numpy(), optimal[x].to_numpy()))

cutoff_comparisons = foo_reduce(cutoff_comparisons)
cutoff_comparisons_high = foo_reduce(cutoff_comparisons_high)
cutoff_comparisons_high_timed = foo_reduce(cutoff_comparisons_high_timed)
plt.plot(np.linspace(-1,0,20), cutoff_comparisons, label='optimal')
plt.plot(np.linspace(-1,0,20), cutoff_comparisons_high, label='high')
plt.plot(np.linspace(-1,0,20), cutoff_comparisons_high_timed, label='high_timed')
plt.legend()
plt.savefig('comparative_plots/filtered_number.png')

  
