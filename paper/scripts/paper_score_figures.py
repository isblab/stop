import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def score_extractor(path, heading):
    df = pd.read_csv(path, sep=' ')
    a = [df[x].to_list() for x in df.columns if heading in x]
    return np.mean(np.array(a), axis=0)


def foo_score_parser(path):
    with open(path) as f:
        rd = f.read().split('\n')
        rd = np.array(list(map(float, [x for x in rd if len(x) > 0])))
    return rd


actin_headings = ['Total_Score', 'ExcludedVolumeSphere_None', 'GaussianEMRestraint_None', 'SAXSRestraint_Score',
                  'CrossLinkingMassSpectrometryRestraint_Data_Score']
gtusc_headings = ['Total_Score', 'CrossLinkingMassSpectrometryRestraint_Data_Score',
                  'ExcludedVolumeSphere_']

for heading in actin_headings:
    actin_path = '../imp_systems/actin_tropomyosin/analysis/final_runs_filtered'
    actin_optimal = score_extractor(f'{actin_path}/optimal/good_scoring_models/model_ids_scores.txt', heading)
    actin_high = score_extractor(f'{actin_path}/high/good_scoring_models/model_ids_scores.txt', heading)
    actin_high_timed = score_extractor(f'{actin_path}/high_timed/good_scoring_models/model_ids_scores.txt', heading)

    plt.figure()
    plt.hist(actin_optimal, bins=100, alpha=0.2, color='red')
    plt.hist(actin_optimal, bins=100, histtype='step', color='red', label='optimal')
    plt.hist(actin_high, bins=100, alpha=0.3, color='blue')
    plt.hist(actin_high, bins=100, histtype='step', color='blue', label='high')
    plt.hist(actin_high_timed, bins=100, alpha=0.2, color='green')
    plt.hist(actin_high_timed, bins=100, histtype='step', color='green', label='timed-high')
    plt.legend()
    plt.savefig(f'../paper_imp_figures/actin_gsm_{heading}.svg')
    plt.close()

for heading in gtusc_headings:
    gtusc_10x_path = '../imp_systems/gtusc_spc110_10x/analysis/final_runs_filtered'
    gtusc_optimal = score_extractor(f'{gtusc_10x_path}/optimal/good_scoring_models/model_ids_scores.txt', heading)
    gtusc_med_high = score_extractor(f'{gtusc_10x_path}/high/good_scoring_models/model_ids_scores.txt', heading)
    gtusc_med_low = score_extractor(f'{gtusc_10x_path}/low/good_scoring_models/model_ids_scores.txt', heading)

    plt.figure()
    plt.hist(gtusc_optimal, bins=100, alpha=0.2, color='red')
    plt.hist(gtusc_optimal, bins=100, histtype='step', color='red', label='optimal')
    plt.hist(gtusc_med_high, bins=100, alpha=0.3, color='blue')
    plt.hist(gtusc_med_high, bins=100, histtype='step', color='blue', label='med-high')
    plt.hist(gtusc_med_low, bins=100, alpha=0.2, color='green')
    plt.hist(gtusc_med_low, bins=100, histtype='step', color='green', label='med-low')
    plt.legend()
    plt.xlabel('Restraint Score')
    plt.savefig(f'../paper_imp_figures/gtusc_gsm_{heading}.svg')
    plt.close()

# all models
actin_path = '../imp_systems/actin_tropomyosin/modeling/final_runs'
actin_optimal = foo_score_parser(f'{actin_path}/optimal_total_scores_all_models.txt')
actin_high = foo_score_parser(f'{actin_path}/high_total_scores_all_models.txt')
actin_low = foo_score_parser(f'{actin_path}/low_total_scores_all_models.txt')
actin_high_timed = foo_score_parser(f'{actin_path}_timed/high_total_scores_all_models.txt')
actin_low_timed = foo_score_parser(f'{actin_path}_timed/low_total_scores_all_models.txt')

maxval = np.percentile(actin_optimal, 99)
minval = min(
    actin_optimal.tolist() + actin_high.tolist() + actin_low.tolist() + actin_high_timed.tolist() + actin_low_timed.tolist())

plt.figure()
plt.hist(actin_optimal[actin_optimal < maxval], bins=100, alpha=0.2, color='red')
plt.hist(actin_optimal[actin_optimal < maxval], bins=100, histtype='step', color='red', label='optimal')
plt.hist(actin_high[actin_high < maxval], bins=100, alpha=0.2, color='blue')
plt.hist(actin_high[actin_high < maxval], bins=100, histtype='step', color='blue', label='high')
plt.hist(actin_low[actin_low < maxval], bins=100, alpha=0.2, color='green')
plt.hist(actin_low[actin_low < maxval], bins=100, histtype='step', color='green', label='low')
plt.hist(actin_low_timed[actin_low_timed < maxval], bins=100, alpha=0.2, color='orange')
plt.hist(actin_low_timed[actin_low_timed < maxval], bins=100, histtype='step', color='orange', label='timed-low')
plt.hist(actin_high_timed[actin_high_timed < maxval], bins=100, alpha=0.2, color='olive')
plt.hist(actin_high_timed[actin_high_timed < maxval], bins=100, histtype='step', color='olive', label='timed-high')
plt.legend()
plt.xlim(minval, maxval)
plt.savefig(f'../paper_imp_figures/actin_all_total_score.svg')
plt.close()

gtusc_path = '../imp_systems/gtusc_spc110_10x/scripts/sample/final_runs'
gtusc_optimal = foo_score_parser(f'{gtusc_path}/optimal_total_scores_all_models.txt')
gtusc_med_high = foo_score_parser(f'{gtusc_path}/high_total_scores_all_models.txt')
gtusc_med_low = foo_score_parser(f'{gtusc_path}/low_total_scores_all_models.txt')
gtusc_path = '../imp_systems/gtusc_spc110_100x/scripts/sample/final_runs'
gtusc_high = foo_score_parser(f'{gtusc_path}/high_total_scores_all_models.txt')
gtusc_low = foo_score_parser(f'{gtusc_path}/low_total_scores_all_models.txt')

maxval = np.percentile(gtusc_optimal, 99.7)
minval = min(
    gtusc_optimal.tolist() + gtusc_med_low.tolist() + gtusc_low.tolist() + gtusc_med_high.tolist() + gtusc_high.tolist())

plt.figure()
plt.hist(gtusc_optimal[gtusc_optimal < maxval], bins=200, alpha=0.2, color='red')
plt.hist(gtusc_optimal[gtusc_optimal < maxval], bins=200, histtype='step', color='red', label='optimal')
plt.hist(gtusc_med_high[gtusc_med_high < maxval], bins=200, alpha=0.2, color='blue')
plt.hist(gtusc_med_high[gtusc_med_high < maxval], bins=200, histtype='step', color='blue', label='med-high')
plt.hist(gtusc_med_low[gtusc_med_low < maxval], bins=200, alpha=0.2, color='green')
plt.hist(gtusc_med_low[gtusc_med_low < maxval], bins=200, histtype='step', color='green', label='med-low')
plt.hist(gtusc_low[gtusc_low < maxval], bins=200, alpha=0.2, color='orange')
plt.hist(gtusc_low[gtusc_low < maxval], bins=200, histtype='step', color='orange', label='low')
plt.hist(gtusc_high[gtusc_high < maxval], bins=200, alpha=0.2, color='olive')
plt.hist(gtusc_high[gtusc_high < maxval], bins=200, histtype='step', color='olive', label='high')
plt.xlim(minval, maxval)
plt.legend()
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.savefig(f'../paper_imp_figures/gtusc_all_total_score.svg')
plt.close()
