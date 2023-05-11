# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

EXTERNAL_OPEN_PATH = '../exports/Nelly.h5ad'
amdata = ad.read_h5ad(EXTERNAL_OPEN_PATH, backed='r')

# %%
# IMPORTS

sns.scatterplot(amdata.var, x='acc', y='bias', hue='status')
