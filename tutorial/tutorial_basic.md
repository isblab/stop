# Basic StOP tutorial for Integrative Modeling on IMP

We shall be using the actin-tropomyosin complex for this tutorial. To begin, first familiarize yourself with the usual IMP workflow, a model script and the basics of IMP modeling. You can have a look at the [actin tutorial](https://integrativemodeling.org/tutorials/actin/) for that.

### Setting the stage
Firstly, we download the scripts and the input data from the [GitHub repository](https://github.com/salilab/actin_tutorial). We primarily need two folders, `data` and `modeling` for this tutorial. Next, make sure you have IMP installed and have installed all the requirements for StOP (`pip install -r requirements.txt`). Finally, copy the four StOP files (`main.py`, `utils.py`, `analyzer.py`, `optimizer.py`) in the folder `modeling` (so that we do not have to keep adding complicated paths to access the files. Otherwise, you can keep these files anywhere you want as long as you modify the paths appropriately). Finally, create a folder called `optimization_data` which stores all the output from StOP.

### Deciding what to optimize
We shall consider three metrics here, and for each, we need to create a regex string which matches all the relevant stat-file fields for the particular metric. We also need an appropriate target range for each of the metrics. 
1. `rb1_acceptance`: This is the rigid-body MC acceptance ratio for tropomyosin. (regex: `MonteCarlo_Acceptance_P29246`, target-range: `[0.3, 0.5]`)
2. `rb2_acceptance`: This is the rigid-body MC acceptance ratio for actin-gelsolin. (regex: `MonteCarlo_Acceptance_P29248`, target-range: `[0.3, 0.5]`)
3. `bead_acceptance`: This is the flexible beads MC acceptance ratio. (regex: `MonteCarlo_Acceptance_BallMover`, target-range: `[0.4, 0.5]`)

We have kept a tighter target range for flexible beads since it is easier to optimize in comparison to the other two.

Next, we specify our five parameters, and for each, the metrics they affect. We also need to specify the initial input domain of the parameters (which we have arbitrarily selected here such that the optimization finishes. In practice, you can choose the widest range which is plausible for the modeling setup at hand)

1. `tropo_rb_mtrans`: Maximum translation for the tropomyosin rigid body. (metrics affected: `rb1_acceptance`, input domain: `[0.1,1]`)
1. `tropo_rb_mrot`: Maximum rotation for the tropomyosin rigid body. (metrics affected: `rb1_acceptance`, input domain: `[0.1,0.5]`)
1. `actgel_rb_mtrans`: Maximum translation for the actin-gelsolin rigid body. (metrics affected: `rb2_acceptance`, input domain: `[0.1,1]`)
1. `actgel_rb_mrot`: Maximum rotation for the actin-gelsolin rigid body. (metrics affected: `rb2_acceptance`, input domain: `[0.1,0.5]`)
1. `bead_mt`: Maximum translation for the flexible beads. (metrics affected: `bead_acceptance`, input domain: `[2,5]`)


### Creating the param file
We now need to create the main input file to feed to StOP which contains all the details needed for the optimization run. The format for this file is `<name of the option> : <value of the option>` on different lines for different options. There may be more than one "value" associated to a particular option.

There are some compulsory options. These include specifying the metrics, the parameters and the command to run.
We begin by specifying the former two.

```
METRIC : rb1_acceptance : 0.3,0.5 : MonteCarlo_Acceptance_P29246
METRIC : rb2_acceptance : 0.3,0.5 : MonteCarlo_Acceptance_P29248
METRIC : bead_acceptance : 0.4,0.5 : MonteCarlo_Acceptance_BallMover
PARAM : tropo_rb_mtrans : rb1_acceptance : 0.1,1
PARAM : tropo_rb_mrot : rb1_acceptance : 0.1,0.5
PARAM : actgel_rb_mtrans : rb2_acceptance : 0.1,1
PARAM : actgel_rb_mrot : rb2_acceptance : 0.1,0.5
PARAM : bead_mt : bead_acceptance : 2,5
``` 
As you can see, each `METRIC` option takes 3 "values", the name of the metric, the target range (comma separated) and the search string to find the relevant fields under the stat file.
Each `PARAM` option takes 3 "values", the name of the parameter, the metrics affected by the parameter (comma separated) and the input domain (comma separated).

To write the `COMMAND` option, we need to know where IMP is installed on the machine. For the tutorial, let us call that the `IMP_PATH`. We also need the modified IMP script to run. We shall be modifying the `modeling_manual.py` script and hence, we will use that script in our command specification.

```
COMMAND : IMP_PATH/setup_environment.sh python -u modeling_manual.py
```

There are several other options that can be specified in the file. Most of them have defaults and are optional. However, it is a good idea to explicitly state them since many of the defaults may not be optimal for the particular machine on which the optimization is being run. For details, read the ReadMe on this repository.

The options that we add here will tell StOP to output all the optimization data to a folder called `optimization_data` that we created earlier. We shall use 5 repeats for each parameter combination, and will be running for 6000 frames. The `m(1)` and `m(2)` values depend on the machine and the expected runtime, but we shall keep this to 8 and 4 respectively. These need to be carefully chosen to maximally utilize the available CPUs.

Have a look at the `param_file` to see the final completed file.

### Modifying the modeling script
First, we need to accept the parameter inputs from the command line. StOP feeds the parameters in the same order as they appear on the input options file, followed by an extra entry specifying the output path which needs to be fixed when running `IMP.pmi.macros.ReplicaExchange0`. 

```
import sys
tropo_rb_mtrans, tropo_rb_mrot, actgel_rb_mtrans, actgel_rb_mrot, bead_mt = map(float, sys.argv[1:6])
output_path_from_cl = sys.argv[6]
```

Next, we put these parameters wherever we need to specify the corresponding quantities in the modeling script. We also need to set the number of frames (to 5000 in this tutorial) and the output path in the `ReplicaExchange0` macro. The modified script is present in the `tutorial` folder along with the original script for comparison (run `diff` on most linux systems to easily compare the two files)

### Running StOP
Next step is to run StOP on our setup. Navigate to the `modeling` folder in a terminal (where we had put the StOP files). Now run `python -u main.py param_file` and wait for StOP to finish. You will see progress bars printed on the terminal. This may take a long time to finish, may even be greater than a couple of days based on the `m` that we set. For smaller systems, this would be faster. 

### Looking at the report
StOP outputs a report on successful completion that is stored in `logs` along with a few other plots/logs/information. The report will look something like `report.txt` present in the `tutorial` folder here. It shows that all the parameter groups were successful, i.e. all the metrics were successfully optimized. It also give the parameter values for which the metrics were found to lie in the target range, and the corresponding metric values with the standard deviations. Based on the plotting level set, `logs` also contains multiple plots.