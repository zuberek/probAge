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

site_indexes = amdata[amdata[amdata.obs.saturating_std].obs.r2.sort_values().index].obs.tail(12).index
axs = plot.tab(site_indexes)
for i, site_index in enumerate(site_indexes):
    sns.scatterplot(x=amdata.var.age, y= amdata[site_index].X.flatten(), ax=axs[i])
    axs[i].set_ylim(0,1)