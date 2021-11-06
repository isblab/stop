# StOP: Stochastic Optimization of Parameters for IMP

![main workflow](https://github.com/isblab/stop/actions/workflows/pytest.yml/badge.svg)

![5z](https://user-images.githubusercontent.com/8314735/140460073-3093167c-dbc4-4560-bbb3-6e6c74d79124.png)

## About
StOP is a gradient-free, parallel, global, stochastic, multi-objective optimization algorithm primarily written to optimize the MCMC (Markov Chain Monte Carlo) sampling parameters for the Integrative Modeling Platform, [IMP](https://integrativemodeling.org). 
The code can be adapted to any suitable optimization problem.

### Requirements
Apart from `python3` and a `linux` system, the code requires the following python modules: `scipy`, `numpy`, `matplotlib`, `tqdm`. You can install all of them using `pip install -r requirements.txt`. While the code is `windows`-compatible for most part, multiple `subprocess` spawning unreliably ends up with a memory error when `max_np > 1`. This is currently an open issue.

### Installation and General Usage:
1. Make sure all the necessary modules are installed
2. Clone the repository or copy the files `main,py`, `utils.py`, `optimizer.py` and, optionally, depending on your usage, `analyzer.py` to a directory of your choosing
3. Create an input file with runtime options and the parameter/metric inputs (described below)
4. `python main.py YOUR_INPUT_FILE`

### Overall workflow
For an IMP-specific tutorial, see [here](https://github.com/isblab/stop/blob/main/tutorial/tutorial_basic.md) for a basic tutorial. More tutorials upcoming!

StOP requires all the input information to be presented as an input file (see below for the format). It requires two other scripts. One script, the IMP modeling script (which represents the metric function) needs to be specified in form of a command to be run at the different parameter values specified in the input file. Another is the optional custom analysis script for which the default analysis script is present in the repository. The algorithm proceeds by parallelizing several invocations of the modeling script and then calling the analysis script on the outputs.

## Input file format
Each line in the input file represents a `KEY : VALUE1 : VALUE2 ...` binding where `KEY` specifies the name of the option and one or more `VALUE`s specify the value(s) of the option or its attributes.

#### Compulsory Options
One or more of the following options must be present in a valid input file. You should replace all values in `<>` with appropriate values for your use case.

1. `METRIC : <name> : <low>,<high> : <regex_search_string>` (The target range of the metric is set as `[<low>, <high>]`. The search string is matched to all the keys of the *stat_file* and the *replica_stat_file* and all matching keys are averaged to get the value of the metric. See analysis details below.)
2. `PARAM : <name> : <metric1>,<metric2> : <low>,<high>` (The third field contains all the metrics affected by this parameter as a comma-separated list. The initial search range (input domain) of the parameter is set as `[<low>, <high>]`)
3. `COMMAND : <command_to_run>` (The command takes the parameters as input from the command line. The CLI arguments are the parameter values presented in the order they are specified in the input file followed by a path to the output folder where the command should save the output files to be analyzed by the analysis script. The output folder is typically set by changing the output of the replica-exchange macro in pmi)

### Optional options
| Option format | Allowed values and default | Description |
|:-------------:|:-------:|:-------:|
|`max_np : <value>`| any integer (default = `os.cpu_count()`) | maximum_number of subprocesses spawned in parallel (note this is in addition to the 3 background processes)|
|`max_depth : <value>`|any integer (default = 5)|maximum allowable depth of the DFS tree|
|`repeat : <value>`|any integer (default = 3)|number of times to rerun the script at the same parameter values to average the stochastic variation|
|`m_<n>d : <value>`|any integer|specify the `m(n)` as specified in the manuscript. One can individually set the value of `m` for different `n` by using multiple `m_<n>d` keys with different `n`|
|`path : <value>`|a valid file path|a path to store all the optimization runtime data in. The IMP output will be stored under folders named `output_<number>` within this path|
|`verbosity : <value>`|0, 1, 2| 0 -> 2 increasing verbosity of the progress|
|`plotting : <value>`|0, 1, 2| 0 -> 2 increasing level of plotting|
|`n_per_command : <value>`|any integer| Number of processes spawned per command invocation. Relevant for `mpirun` based replica exchange. Allows for a better progress bar.|
|`n_frames : <value>`|any integer| Number of frames per IMP run. Allows for a better progress bar.|
|`max_wait : <value>`|any integer (default=60)|Number of seconds to wait before polling each running subprocess for the current status. Also controls the rate of logging|

### Analysis Details

The user submitted regex strings are matched across all the keys in stat-file, and on failing to match any keys, the stat-replica-file. Next, the values for these matched headers are extracted. The total score of the run is then checked for equilibriation by comparing the mean of the final quarter of frames to the penultimate quarter. If either of the means are outside 2SD of either of the quarters, the run is classified as unequilibriated. Next, the matched headers are extracted for the final half of the runs. These are assumed to be cumulative statistics and a correction is applied to them to fix this. Next, they are averaged across the matched headers, across the frames and across replicas (if applicable).

### Upcoming changes

1. More tutorials (including a tutorial to optimize replica exchange maximum temperature)
2. Better documentation of the options above
3. More comprehensive tests
4. Sampling the inner regions of hyper-rectangles by allowing diagonal iterable ranges

### Manuscript

Pasani S, Viswanath S. A Framework for Stochastic Optimization of Parameters for Integrative Modeling of Macromolecular Assemblies. Life. 2021; 11(11):1183. https://doi.org/10.3390/life11111183

The relevant paper can be found [here](https://doi.org/10.3390/life11111183). Paper-related scripts and the explanation for the reproduction of figures can be found at the zenodo link [here](https://doi.org/10.5281/zenodo.5521444
).