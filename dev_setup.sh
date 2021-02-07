#!/usr/bin/env

python3 -m venv venv
source venv/bin/activate
pip3 install numpy numexpr tqdm pygam scikit-learn
pip3 install coverage line_profiler twine wheel sphinx_rtd_theme ipython ipdb
pip3 install jedi==0.17.2
