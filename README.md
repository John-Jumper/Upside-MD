## Installation

#### Dependencies

Compile dependencies

  * CMake, 2.8+
  * C++11 compiler, such as GCC 4.8+
  * HDF5, 1.8+ with high-level interface compile option
  * Eigen Matrix Library, 3.0+

Python dependencies.  Depending on your use-case for Upside, you may not need
all of these libraries.  The error message in case of a missing library should
be clear enough to figure out.

  * Python 2.7
  * Numpy
  * Scipy (sometimes needed for `upside_config.py`, depending on options)
  * PyTables (needed for reading HDF5 files from Python)
  * Prody (needed for reading .pdb files)
  * Pandas (needed for `predict_chi1.py`)
  * MDTraj (optional but capable of loading and analyzing Upside trajectories)

#### Obtaining the source code and parameter files

The source code is obtained from this repository.  The parameter files must
be obtained separately.  We are working on some licensing issues that currently
prevent including the parameters with the source code.

## Acquiring specific versions of Upside to reproduce a paper

Each paper that uses Upside should create a git tag that indicates the precise
git version of Upside needed to reproduce the results of the paper.  The
following paper tags have been created.

  * `sidechain_paper` -- "Maximum-likelihood, self-consistent side chain free
    energies with applications to protein molecular dynamics"
  * `trajectory_training_paper` -- "Trajectory-Based Parameterization of a
    Coarse-Grained Forcefield for High-Thoughput Protein Simulation"

To get the precise version of Upside needed for the side chain free energy
paper, just do `git checkout sidechain_paper`.  Note that you should ignore the
rest of this ignore, and consult `README.md` after checking out the tagged
version.  The compile or run instructions may be different between the versions.

External users of Upside are encourage to tag the version used in their work
and the tag will be added to the main Upside repository (feel free to open a
GitHub issue for this).  This will also help the Upside team keep track of
published results using Upside.  If modifications of Upside are required,
please create a branch and open a GitHub pull request.  The branch will be
preserved in this repository for reproducibility even if it is not merged into
the master branch.

#### Compiling Upside

This guide assumes that you have all of the compile dependencies satisfied.  In
particular, the HDF5 high-level must be specifically enabled by a configuration
switch when installing HDF5.  Please check with your system administrator to
ensure it is installed.

This install guide assumes that you are using Linux or OS X.  Upside should
install and run on Windows as well, but this has not been tested.

The build system uses CMake (www.CMake.org) to find dependencies.  Please
change to the `obj/` directory of the upside directory and execute the following
commands.

    rm -rf ../obj/* && cmake ../src && make -j

Due to a quirk in the current handling of parameters, Upside must be recompiled
to handle the 10A cutoff parameter files.  If you want to use the 10A cutoff
parameters for either chi1 prediction or molecular dynamics, please compile with
the following commands.

    rm -rf ../obj/* && CXXFLAGS=-DPARAM_10A_CUTOFF cmake ../src && make -j

After these commands execute successfully, the `obj/` directory will contain
the `upside` executable and the `libupside.so` shared library (exact name of
shared library may depend on operating system).

#### Predicting chi1 rotamer states

After compiling the code, you may perform chi1 prediction on a PDB file.  

    upside/py/predict_chi1.py --sidechain-params /path/to/params.h5 input.pdb output.chi

As noted above, you must compile with `CXXFLAGS=-DPARAM_10A_CUTOFF` to use the
10A cutoff parameters.  The output will be of the form

    residue restype chain resnum chi1_prob0 chi1_prob1 chi1_prob2 chi1_from_input_file
    0       ILE     A     1      0.0000     0.0003     0.9997     -59.7
    1       VAL     A     2      0.9986     0.0014     0.0001     54.7
    2       GLY     A     3      1.0000     0.0000     0.0000     nan

The residue is 0-indexed along the chains, but the resnum is taken from the
original PDB file. 

The Upside chi1 predictions are expressed as probabilities
for each of the three chi1 rotamers, representing the ranges [0,120),
[120,240), and [240,360), respectively.  The probabilities are directly the
1-body marginal probabilities from the belief propagation, summed over the chi2
states where necessary.

The `chi1_from_input_file` field returns the chi1 angle in the input PDB file.
Any residue without a chi1 angle (either because of the amino acid or just
because it is omitted from the input .pdb) will give NaN.  The chi1 angles from
the input PDB file have no effect on the `chi1_prob` fields.

A note on the timing information that `predict_chi1.py` reports.  The
`predict_chi1.py` executable is rather inefficient, since it is really a
by-product of the MD engine.  For chi1 prediction, we must read the PDB file,
run `upside_config.py` to create a molecular dynamics configuration, start the
Upside MD engine, compute a single MD energy evaluation, read the side chain
marginal probabilities from the MD engine, and then write the output text file.
The "Time to compute" line in the output (which is the evaluation time reported
in the paper) only counts the time involved in the energy evaluation and
reading the marginal probabilities.  This time is chosen because it represents
the repeated work that is involved in setting up and solving the side chain
interactions for a new backbone position without the one time costs of
converting the PDB to an Upside configuration, booting the Upside engine, or
writing the output.  This time is a bit slower than a typical MD force
evaluation for the same structure in part because Upside must calculate pair
lists for the first time.  

#### Running molecular dynamics

The following sections illustrate running simple molecular dynamics simulations
with Upside.

### Converting a PDB file to Upside input

Starting from a PDB file, we create FASTA, position, and side chain files using

    upside/py/PDB_to_initial_structure.py /path/to/input.pdb output_basename

which will create `output_basename.fasta` containing the FASTA sequence,
`output_basename.initial.pkl` containing a Python pickle file of the
coordinates of the N, CA, and C atoms of the backbone, and
`output_basename.chi` containing the chi1 and chi2 angles for the side chains.
The .chi file is not needed for MD simulation, unless you want to run with
inflexible side chain rotamers.  `PDB_to_initial_structure.py` will fail if there
are breaks in the chain, as chain breaks will cause the MD to be invalid.  See
`--help` for information on how to choose the PDB model or chains to include.

From the output of `PDB_to_inital_structure.py`, we can create an Upside
configuration, `simulation.up`.  The following instructions are for a typical
folding or conformational dynamics simulation.

    upside/py/upside_config.py --output                   simulation.up \
                               --fasta                    output_basename.fasta \
                               --initial-structure        output_basename.initial.pkl \
                               --hbond-energy             $(cat upside/parameters/ff_1/hbond) \
                               --dynamic-rotamer-1body    \
                               --rotamer-placement        upside/parameters/ff_1/sidechain.h5 \
                               --rotamer-interaction      upside/parameters/ff_1/sidechain.h5 \
			       --environment              upside/parameters/ff_1/environment.h5 \
                               --rama-library             upside/parameters/common/rama.dat \
			       --rama-sheet-mixing-energy $(cat upside/parameters/ff_1/sheet) \
                               --reference-state-rama     upside/parameters/common/rama_reference.pkl

If `--initial-structure` is omitted, the simulation will be initialized with
random (possibly-clashing) Ramachandran angles, which is useful for de novo
structure prediction.

The output `simulation.up` is an HDF5 file, and the simulation configuration is
written in the "/input" group within the `.up` file.  The `upside_config.py`
copies all of the parameters into its output, so that only `simulation.up` is
needed to run the simulation.  

## Constant temperature simulation

A simple, constant-temperature simulation may be run with 

    upside/obj/upside --duration 1e7 --frame-interval 1e2 --temperature 0.5 --seed $RANDOM simulation.up

Note that the simulation results are written into the "/output" group of the
provide `simulation.up` file.  This ensures that simulation output is stored in
the same file as the input, aiding in later reproducibility of the simulation.
The temperature, duration, and frame-interval are in natural units.  The times
should not be interpreted as picoseconds.  We are still investigating the
precise relationship of Upside time to experimental folding time, but the time
relationship may be conformation-dependent since the Upside backbone moves in a
smoother energy landscape than standard MD due to the side chain model. 

### Replica exchange simulation

The equilibration time of Upside simulations is highly temperature-dependent.
Typically, folding studies in Upside start with a replica exchange simulation
to establish the temperature range of the melting transition.  Hamiltonian and
temperature replica exchange are handled in the same manner in Upside.  First,
use `upside_config.py` repeatedly to create `N` configurations (or just copy
the output of a single `upside_config.py` run).  To run the replica exchange on
five configurations,

    upside/obj/upside --duration 1e7 --frame-interval 1e2 \
                      --temperature 0.5,0.53,0.56,0.60,0.64 \
                      --swap-set 0-1,2-3 \
                      --swap-set 1-2,3-4 \
                      --replica-interval 20 \
                      --monte-carlo-interval 5 \
                      --seed $RANDOM \
                      config_0.up config_1.up config_2.up config_3.up config_4.up

The multiple simulations are parallelized using OpenMP threads on a single
machine.  On the SLURM scheduler, jobs should be launched with `sbatch --ntasks
1 --cpus-per-task 5` for a five simulation replica exchange job.  The
`--replica-interval` controls the frequency at which replica exchange swaps are
attempted.  The `--swap-sets` arguments are *non-overlapping* pairs of replica
swaps that are attempted simultaneously.  For a linear chain of temperatures,
there should always be two swap sets to ensure ergodicity of swaps without
having any overlapping pairs.

The `--monte-carlo-interval` controls the frequency at which pivot moves, a
discrete change in the phi and psi angles of a single, randomly-chosen residue,
are attempted.  These Monte Carlo pivot moves are accepted with a Metropolis
criterion so that the Boltzmann ensemble of simulation is unchanged by pivot
moves.  Typically, pivot moves are highly advantageous in replica exchange
simulation because pivot moves have high acceptance rates in high-temperature
unfolded conformations, greatly speeding the sampling of unfolded states.
Pivot moves have very low acceptance in collapsed conformations but add very
little to the computational time.  The faster sampling in the unfolded states
feeds conformational diversity to the lower temperatures in replica exchange.
Monte Carlo moves are also permitted in constant temperature simulation.  If
`--monte-carlo-interval` is not provided, pivot moves are never attempted.

The output of replica exchange is written into the "/output" group of each of
the input configurations.  The replicas are written by temperature/Hamiltonian,
so that in the example above, `config_0.up` contains all of the simulation
frames at the temperature 0.5.  This means that in the limit of infinite sampling, 
`config_0.up` will contain samples from a Boltzmann ensemble at temperature 0.5 but 
the trajectory will be discontinuous due to replica swapping.

#### Simulation analysis and visualization

The contents of a `.up` file can be listed with

    upside/py/attr_overview.py simulation.up

which included copies of the full command lines used to invoke both
`upside_config.py` and `upside`.  

To load the simulation in VMD, first convert to a trajectory format
that VMD can read.  This can be done with

    upside/py/extract_vtf simulation.up simulation.vtf

and the `.vtf` file can be read natively by VMD. Alternatively, the MDTraj
library described below can read `simulation.up` directly, either to visualize
in an IPython notebook or to convert to another format.  In either case, amide
hydrogen and carbonyl oxygen are added as atoms in the trajectory to aid
structure viewers.

### Using MDTraj

To load the Upside trajectory as an MDTraj Trajectory object, use the
`mdtraj_upside.py` library in the Upside distribution.  

    import sys
    sys.path.append('upside/py')
    import mdtraj_upside as mu
    traj = mu.load_upside_traj('simulation.up')

When the trajectory is loaded, the atoms H, O, and CB will be added to the
trajectory.
