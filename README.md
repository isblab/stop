# StOP: Stochastic Optimization of Parameters for IMP

## About
StOP is an gradient-free, parallel, global optimization framework primarily written to optimize the MCMC (Markov Chain Monte Carlo) sampling parameters for the Integrative Modeling Platform. It utilizes a parallel DFS-based search similar to MCS but adapted to a range-finding setup.
The code can be adapted to any suitable optimization problem outside of IMP too.
The relevant paper can be found here (link) and the paper related scripts, plots and the explanation is in the `paper` folder.

### Requirements
Apart from `python3` and a `linux` system, the code requires the following python modules: `scipy`, `numpy`, `matplotlib`, `tqdm`. While the code is `windows`-compatible for most part, multiple `subprocess` spawning unreliably ends up with a memory error when `max_np > 1`. This is currently an open issue.

### Installation and General Usage:
1. Make sure all the necessary modules are installed
2. Clone the repository or copy the files `main,py`, `utils.py`, `optimizer.py` and (optionally, depending on your usage) `analyzer.py` to a directory of your choosing
3. Create an input file with runtime options and the parameter/metric inputs (described below)
4. `python main.py YOUR_INPUT_FILE`

### Overall workflow
For an IMP-specific tutorial, see here.
StOP requires all the input information to be presented as an input file (see below for the format). It requires two other scripts. One script, the IMP modeling script (which represents the metric function) needs to be specified in form of a command to be run at the different parameter values in the input file. Another is the optional custom analysis script for which the default analysis script is present in the repository.
The algorithm proceeds by parallelizing several invocations of the modeling script and then calling the analysis script on the outputs.

## Input file format
Each line in the input file represents a `KEY : VALUE1 : VALUE2 ...` binding where `KEY` specifies the name of the option and one or more `VALUE`s specify the value(s) of the option or its attributes.

#### Compulsory Options
One or more of the following options must be present in a valid input file. You should replace all values in `<>` with appropriate values for your use case.

1. `METRIC : <name> : <low>,<high> : <regex_search_string>` (The target range of the metric is set as `[<low>, <high>]`. The search string is matched to all the keys of the stat_file and the replica_stat_file and all matching keys are averaged to get the value of the metric. See the details of analysis below)
2. `PARAM : <name> : <metric1>,<metric2> : <low>,<high>` (The second column contains all the metrics affected by this parameter as a comma-separated list. The initial search range of the parameter is set as `[<low>, <high>]`)
3. `COMMAND : <command_to_run>` (The command takes the parameters as input from the command line. The CLI arguments are the parameter values in the order they are specified in the input file followed by a path to the output folder where the command should save the output files to be analyzed by the analysis script)

### Optional options
| Option format | Allowed values and default | Description |
|:-------------:|:-------:|:-------:|
|`max_np : <value>`| any integer (default = `os.cpu_count()`) | maximum_number of subprocesses spawned in parallel (note this is additional to the 3 background processes)|
|`max_depth : <value>`|any integer (default = 5)|maximum allowable depth of the DFS tree|
|`repeat : <value>`|any integer (default = 3)|number of times to rerun the script at the same parameter values to average the stochastic variation|
|`m_<n>d : <value>`|any integer (see below for default values)|specify the `m(n)` as specified in the manuscript|
|`path : <value>`|a valid file path|a path to store all the optimization runtime data in (including the output)|
|`verbosity : <value>`|0, 1, 2| 0 -> 2 increasing verbosity of the progress|
|`plotting : <value>`|0, 1, 2| 0 -> 2 increasing level of plots|