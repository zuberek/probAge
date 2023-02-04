%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)


wave3 = amdata_src.AnnMethylData('../exports/wave3_linear.h5ad')

DATA_PATH = '../exports/wave1_linear.h5ad'

amdata = amdata_src.AnnMethylData(DATA_PATH)
amdata= ad.read_h5ad(DATA_PATH)

# Add ground truth from wave3
params = ['mean_slope','mean_inter','var_slope','var_inter']
intersection = wave3.obs.index.intersection(amdata.obs.index)
amdata = amdata[intersection]
wave3 = wave3[intersection]
amdata.obs[params] = wave3.obs[params]
amdata = amdata_src.AnnMethylData(amdata)

# Refit to quickly check for differences
true_params = [f'true_{param}' for param in params]
maps = modelling.drift_sites(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
for param in params:
    amdata.sites[f'true_{param}'] = maps[param]

axs = plot.tab(params, 'Hannum - Wave 3 comparison', ncols=2, row_size=5, col_size=8)
for i, param in enumerate(params):
    sns.scatterplot(data=amdata.sites, x=param, y=f'true_{param}', ax=axs[i])

amdata.write_h5ad(DATA_PATH)

# Calculate predictions without any correction
ab_maps = modelling.person_model(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
sns.histplot(x=ab_maps['acc'], y=ab_maps['bias'], cbar=True, ax=plot.row('Before'))

# Infer the offsets
maps = modelling.site_offsets(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata.obs['offset'] = maps['offset']
sns.histplot(amdata.obs.offset)
amdata.obs.sort_values('offset')

# Plot the data correction
ax = plot.row('Top shifted down')
sns.scatterplot(x=wave3.var.age, y=wave3['cg07497042'].X.flatten(), label='wave3',ax=ax)
sns.scatterplot(x=amdata.var.age, y=amdata['cg07497042'].X.flatten(), label='hannum', ax=ax)
sns.scatterplot(x=amdata.var.age, y=amdata['cg07497042'].X.flatten()-amdata['cg07497042'].obs.offset.values[0], label='hannum (corr)', ax=ax)

ax = plot.row('Top shifted up')
sns.scatterplot(x=wave3.var.age, y=wave3['cg05404236'].X.flatten(), label='wave3',ax=ax)
sns.scatterplot(x=amdata.var.age, y=amdata['cg05404236'].X.flatten(), label='hannum', ax=ax)
sns.scatterplot(x=amdata.var.age, y=amdata['cg05404236'].X.flatten()-amdata['cg05404236'].obs.offset.values[0], label='hannum (corr)', ax=ax)

# Apply the offset and refit acceleration and bias
offset = np.broadcast_to(amdata.obs.offset, shape=(amdata.n_participants, amdata.n_sites)).T
amdata.X = amdata.X - offset
ab_maps = modelling.person_model(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
sns.histplot(x=ab_maps['acc'], y=ab_maps['bias'], cbar=True, ax=plot.row('After'))

amdata.var['raw_acc'] = ab_maps['acc']
amdata.var['raw_bias'] = ab_maps['bias']
amdata.var['corr_acc'] = ab_maps['acc']
amdata.var['corr_bias'] = ab_maps['bias']

sns.histplot(data=amdata.var, x='raw_acc', y='corr_acc')
sns.lineplot(x=[-1,1], y=[-1,1])

sns.histplot(data=amdata.var, x='raw_bias', y='corr_bias')
sns.lineplot(x=[-0.1,0.1], y=[-0.1,0.1])

