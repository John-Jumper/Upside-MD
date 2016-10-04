## Installation

#### Dependencies

Compile Dependencies

  * CMake, 2.8+
  * GCC C++ compiler, 4.8+
  * HDF5, 1.8+
  * Eigen Matrix Library, 3.0+


Dependences for generating config files

  * Python 2.6+
  * Numpy python library
  * Scipy python library
  * PyTables python library
  * Prody library for converting PDB files


dependences.  Note that the `module` command must be re-executed every time you
login or put in your `.bashrc`.  The `easy_install` commands only need to be
executed once.

    module add git cmake gcc hdf5 python eigen
    easy_install --user numexpr cython cvxopt prody
    easy_install --user tables

#### Obtaining the source code and parameter files

The source code is obtained from this repository.  The parameter files must
be obtained separately.  We will shortly add a separate upside-parameters repository
to contain them.

#### Compiling the code
If you are running on midway, godzilla, Ubuntu, Debian, or OS X, try
`initial_setup.bash` in the upside directory.

Note that the build system uses CMake (www.CMake.org) to find dependencies.  If
you just want to build and run the code, try `get_new_version.bash` in the `upside`
directory.

After executing these commands, there will be an `upside` binary in the `obj/`
directory.  The configuration generator `upside_config.py` will still be in the `src/`
directory because python files do not require compilation.

## Using Upside

TODO: update this section.

#### Generating a configuration file

The program `upside_config.py` generates the input `.h5` file for `upside`.

#### Running a simulation

The program `upside` runs the simulation.  The input `.h5` defines the model
(energy terms and initial position) and the arguments to `upside` define the
parameters to run the simulation (duration, time step, temperature, etc.).  
