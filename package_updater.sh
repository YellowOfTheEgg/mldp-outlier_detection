#!/bin/bash
rm -r venv
virtualenv venv
source venv/bin/activate
#pip install .
python3 -m pip install -e .