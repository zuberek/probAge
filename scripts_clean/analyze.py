# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

amdata = ad.read_h5ad(paths.DATA_FITTED, backed='r')

# %%
# IMPORTS

sns.scatterplot(amdata[:,amdata.var.status=='test'].var, x='acc', y='bias', hue='age', palette='rocket')
sns.scatterplot(amdata[:,amdata.var.status=='control'].var, x='acc', y='bias', marker='x', color='tab:green', label='control')

sns.scatterplot(amdata.var, x='age', y='acc')
sns.scatterplot(amdata.var, x='age', y='bias')
