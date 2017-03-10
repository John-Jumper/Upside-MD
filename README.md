## Installation

#### Dependencies

Compile dependencies

  * CMake, 2.8+
  * C++11 compiler, such as GCC 4.8+
  * HDF5, 1.8+ with high-level interface compile option
  * Eigen Matrix Library, 3.0+

Dependences for generating config files

  * Python 2.7
  * Numpy python library
  * Scipy python library
  * PyTables python library
  * Prody library for reading PDB files

#### Obtaining the source code and parameter files

The source code is obtained from this repository.  The parameter files must
be obtained separately.  We will shortly add a separate upside-parameters repository
to contain them.

## Acquiring specific versions of Upside to reproduce a paper

Each paper that uses Upside should create a git tag that indicates the precise
git version of Upside needed to reproduce the results of the paper.  The
following paper tags have been created.

  * `sidechain_paper` -- "Maximum-likelihood, self-consistent side chain free
    energies with applications to protein molecular dynamics"

To get the precise version of Upside needed for the side chain free energy
paper, just do `git checkout sidechain_paper`.  Note that you should ignore the
rest of this README, and consult README.md after checking out the tagged
version.  The compile or run instructions may be different between the versions.

External users of Upside are encourage to tag the version used in their work
and the tag will be added to the main Upside repository (feel free to open a
GitHub issue for this).  This will also help the Upside team keep track of
published results using Upside.  If modifications of Upside are required,
please create a branch and open a GitHub pull request.  The branch will be
preserved in this repository for reproducibility even if it is not merged into
the master branch.

#### Compiling the code

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
parameters for either chi1 prediction or molecular dynamics, please use the following
commands.

    rm -rf ../obj/* && CXXFLAGS=-DPARAM_10A_CUTOFF cmake ../src && make -j

After these commands execute successfully, the `obj/` directory will contain
the `upside` executable and the `libupside.so` shared library (exact name of
shared library may depend on operating system).

#### Predicting chi1 rotamer states

TODO: update this section.

#### Running molecular dynamics
