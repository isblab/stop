## Overall description of the contents and steps for reproducing the figures

### Example Data
To generate the example data, first run StOP (`python main.py <input file>`) on all the param files in `examples/example_1` and `2` and then run the scripts in the `scripts` folder. The data necessary for plotting is already currently present in the `examples` folder.

### IMP systems Data
The current `imp_systems` folder contains all the bash scripts for modeling and analysis. However, due to space constraints, the `imp_systems` folder with the results of the final runs, the input data needed for modeling and the filtered models can be accessed here (zenodo link) and needs to be extracted in this folder (replcae the current `imp_systems` folder) before running the steps below.
To generate IMP data (all the sets of the runs), all the bash scripts for modeling are present in `actin/modeling` for the actin system and `gtusc_spc110_10x/scripts/sample` (same path under `gtusc_spc110_100x`) for the gtusc system.
Analysis bash scripts are in the `analysis` folder under the system-specific parent folders.
Both the `analysis` and `modeling` bash scripts will generate the results (currently already present in the `imp_systems` folder uploaded on zenodo)

### Figures
`paper_example_figures` contain the example figures used in the paper and `random_metric_functions` contains randomly generated 1D and 2D functions with StOP applied on them.
`paper_imp_figures` contain the figures used in the paper.
Finally, run the `scripts` under the `scripts` folder after all the data is generated to reproduce the figures.

## Details of the bash scripts
1. `all_run.sh` for both the systems is the script with the commands to run StOP and the final runs.
2. `timing_runs.sh` for both the systems is the script with the commands to run the timing runs with the timing information stored in `timing/timing_data.txt`. Note that this bash script needs to be manually killed (`ctrl + C`) after the timing runs are complete since the python script for keeping the time of frames does not automatically quit after the runs are done.
3. `all_run_timed.sh` for the actin system is the script to run the timed runs.
4. `filter.sh` and `filter_final.sh` for both the systems contain the commands to filter based on cross-linking criterion alone, plot the score plots for models passing this criterion along with the mean - 0.25SD cutoffs for the restrains based on the models passing this cutoff, and to get the good-scoring models after applying cutoff thresholds for all the restraints.
5. `exhaustiveness.sh` for both the systems contain the commands to run the sampling exhaustiveness/convergence pipeline

**Note**: While the copies of StOP files are present in all the IMP system folders to make the bash scripting easier, it is sufficient to have a single copy of the files `optimizer.py`, `main.py`, `analysis.py` and `utils.py` and modify the paths in the bash scripts accordingly.

**Note**: Only the good-scoring models, the filtered model-ids and the results of the analysis pipeline (sampling exhaustiveness) are present in the `imp_systems` zenodo link.
