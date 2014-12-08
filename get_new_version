#!/bin/bash

set -e
git pull  # obtain new source code
script_path="`dirname $0`"  # find this script

cd "$script_path/obj"  # go to obj/ subdirectory
make -j4               # compile the code with 4 processors
