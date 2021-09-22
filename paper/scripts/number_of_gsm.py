import matplotlib.pyplot as plt
import numpy as np


def line_counter(path):
    c = -1
    with open(path) as f:
        for i in f:
            c += 1
    return c


actin_path = '../imp_systems/actin_tropomyosin/analysis/final_runs_filtered'
actin_optimal = line_counter(f'{actin_path}/optimal/good_scoring_models/model_ids_scores.txt')
actin_high = line_counter(f'{actin_path}/high/good_scoring_models/model_ids_scores.txt')
actin_low = line_counter(f'{actin_path}/low/good_scoring_models/model_ids_scores.txt')
actin_high_timed = line_counter(f'{actin_path}/high_timed/good_scoring_models/model_ids_scores.txt')
actin_low_timed = line_counter(f'{actin_path}/low_timed/good_scoring_models/model_ids_scores.txt')

gtusc_10x_path = '../imp_systems/gtusc_spc110_10x/analysis/final_runs_filtered'
gtusc_100x_path = '../imp_systems/gtusc_spc110_100x/analysis/final_runs_filtered'
gtusc_optimal = line_counter(f'{gtusc_10x_path}/optimal/good_scoring_models/model_ids_scores.txt')
gtusc_med_high = line_counter(f'{gtusc_10x_path}/high/good_scoring_models/model_ids_scores.txt')
gtusc_med_low = line_counter(f'{gtusc_10x_path}/low/good_scoring_models/model_ids_scores.txt')
gtusc_high = line_counter(f'{gtusc_100x_path}/high/good_scoring_models/model_ids_scores.txt')
gtusc_low = line_counter(f'{gtusc_100x_path}/low/good_scoring_models/model_ids_scores.txt')

plt.figure()
plt.bar(np.arange(5), [actin_high, actin_high_timed, actin_optimal, actin_low_timed, actin_low],
        tick_label=['high', 'timed-high', 'optimal', 'timed-low', 'low'], color='darkred')
plt.xlabel('Types of sampling')
plt.ylabel('Number of good-scoring models')
plt.savefig('../paper_imp_figures/gsm_actin.svg')
plt.close()

plt.figure()
plt.bar(np.arange(5), [gtusc_high, gtusc_med_high, actin_optimal, gtusc_med_low, gtusc_low],
        tick_label=['high', 'med-high', 'optimal', 'med-low', 'low'], color='darkred')
plt.xlabel('Types of sampling')
plt.ylabel('Number of good-scoring models')
plt.savefig('../paper_imp_figures/gsm_gtusc.svg')
plt.close()
