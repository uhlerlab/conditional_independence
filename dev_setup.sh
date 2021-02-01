#!/usr/bin/env

python3 -m venv venv
source venv/bin/activate
pip3 install ipython numpy numexpr ipdb tqdm pygam scikit-learn twine wheel sphinx_rtd_theme
pip3 install coverage
pip3 install jedi==0.17.2
