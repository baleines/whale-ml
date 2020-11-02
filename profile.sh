#! /bin/bash
python -m cProfile -o train.profile train.py
pyprof2calltree -i train.profile -o train.calltree
kcachegrind train.calltree