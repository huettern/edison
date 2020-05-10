#!/bin/bash

# init submodules
git submodule update --init --recursive

# create virtual env
cd audio/
virtualenv -p python3.7 venv

# add paths to virtualenv
echo "export PYTHONPATH=`pwd`:$PYTHONPATH" >> venv/bin/activate
echo "export PYTHONPATH=`pwd`/implement/nnom/scripts:$PYTHONPATH" >> venv/bin/activate

# install packages
source venv/bin/activate
pip install -r requirements.txt

