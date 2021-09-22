import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

low = pd.read_csv('low/good_scoring_models/model_ids_scores.txt', sep=' ')
optimal = pd.read_csv('optimal/good_scoring_models/model_ids_scores.txt', sep=' ')
high = pd.read_csv('high/good_scoring_models/model_ids_scores.txt', sep=' ')

print(f'Number of Models: optimal={len(optimal)}, high={len(high)}, low={len(low)}')

for x in high.columns.to_numpy()[4:]:
    rng = np.mean(optimal[x]) + np.array([-1, 1]) * np.std(optimal[x]) * 3
    a = np.mean(optimal[x]) + np.std(optimal[x])
    b = np.mean(optimal[x]) - np.std(optimal[x])
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.hist(high[x], bins=100, alpha=0.2, label='high', range=rng)
    ax.hist(optimal[x], bins=100, alpha=0.2, label='optimal', range=rng)
    ax.hist(low[x], bins=100, alpha=0.2, label='low', range=rng)
    ax.axvline(a, color='black', linestyle='--')
    ax.axvline(b, color='black', linestyle='--')
    ax.legend()
    
    fig.suptitle(x)
    plt.savefig(f'comparative_plots/comparative_{x}_final_filtering.png')
    plt.close()
