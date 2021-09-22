import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def binary_n(dr):
    return np.ceil(np.log2(dr))


def mary(dr):
    m = np.arange(2, 16)
    return m, np.ceil(np.log2(dr / (m - 1)) / np.log2(m + 1) + 1)


fig, ax = plt.subplots(figsize=(6, 6))
colors = ['#173f5f', '#3caea3', '#f6d55c', '#ed553b']
for dr, col in zip([2, 10, 100, 1000], colors):
    x = np.array([2, 16])
    ax.plot(x, [binary_n(dr), binary_n(dr)], color=col, linestyle='--')
    x, y = mary(dr)
    ax.plot(x, y, color=col)
    ax.scatter(x, y, zorder=10, color=col, marker='^')

ax.set_xlabel('m for m-ary search')
ax.set_ylabel('Number of iterations ($\\propto$ time taken)')

legend_elements = [Line2D([0], [0], color='k', lw=1, linestyle='--', label='Binary'),
                   Line2D([0], [0], marker='^', color='k', label='m-ary')]
legend_elements += [Patch(facecolor=col, label=f'r={dr}') for dr, col in zip([2, 10, 100, 1000], colors)]
ax.legend(handles=legend_elements)
plt.savefig('../paper_example_figures/binary_vs_mary.svg')
