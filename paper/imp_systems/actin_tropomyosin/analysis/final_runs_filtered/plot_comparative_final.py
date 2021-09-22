import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

low = pd.read_csv('low/good_scoring_models/model_ids_scores.txt', sep=' ')
optimal = pd.read_csv('optimal/good_scoring_models/model_ids_scores.txt', sep=' ')
high = pd.read_csv('high/good_scoring_models/model_ids_scores.txt', sep=' ')
high_timed = pd.read_csv('high_timed/good_scoring_models/model_ids_scores.txt', sep=' ')
low_timed = pd.read_csv('low_timed/good_scoring_models/model_ids_scores.txt', sep=' ')

print(f'Number of Models: optimal={len(optimal)}, high={len(high)}, low={len(low)}, high_timed={len(high_timed)}, low_timed={len(low_timed)}')

for x in high.columns.to_numpy()[4:]:
    rng = np.mean(optimal[x]) + np.array([-1, 1]) * np.std(optimal[x]) * 3
    a = np.mean(optimal[x]) + np.std(optimal[x])
    b = np.mean(optimal[x]) - np.std(optimal[x])
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
    plt.savefig(f'comparative_plots/comparative_{x}_final_filtering.png')
    plt.close()
