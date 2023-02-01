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


wave3 = amdata_src.AnnMethylData('../exports/wave3_linear.h5ad', backed='r')

DATA_PATH = '../exports/hannum.h5ad'

amdata = amdata_src.AnnMethylData(DATA_PATH)
amdata= ad.read_h5ad(DATA_PATH)

params = ['mean_slope','mean_inter','var_slope','var_inter']
intersection = wave3.obs.index.intersection(amdata.obs.index)
amdata = amdata[intersection]
wave3 = wave3[intersection]
amdata.obs[params] = wave3.obs[params]

# amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index[:N_SITES]]

maps = modelling.drift_sites(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata.sites[params] = maps[params]

amdata.write_h5ad(DATA_PATH)

ab_maps = modelling.person_model(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']

amdata.var['acc'] = ab_maps['acc']
amdata.var['bias'] = ab_maps['bias']

sns.histplot(data=amdata.var, x='acc', y='bias', cbar=True, ax=plot.row('Wave 1: After'))

maps = modelling.dset_offsets(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata.obs['offset'] = maps['offset']

amdata.obs.offset.hist()

amdata.obs.sort_values('offset')

offset = amdata.obs.offset
offset = np.broadcast_to(amdata.obs.offset, shape=(amdata.n_participants, amdata.n_sites)).T
amdata.X = amdata.X + offset
ax = plot.row('Top shifted: Before')
sns.scatterplot(x=wave3.var.age, y=wave3['cg11142333'].X.flatten(), label='wave3',ax=ax)
sns.scatterplot(x=amdata.var.age, y=amdata['cg11142333'].X.flatten(), label='hannum', ax=ax)
sns.scatterplot(x=amdata.var.age, y=amdata['cg11142333'].X.flatten()-amdata['cg11142333'].obs.offset.values[0], label='hannum (corr)', ax=ax)

wave3.plot.site('cg04875128')
sns.scatterplot(x=amdata.participants.age, y=amdata['cg04875128'].X.flatten())

sns.histplot(offset[:,0])