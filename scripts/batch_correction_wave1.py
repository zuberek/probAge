
%load_ext autoreload
%autoreload 2

from cProfile import label
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
from functools import partial
import arviz as az

import pickle

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

amdata_path = '../exports/wave1_meta.h5ad'
site_info_path = '../exports/wave3_acc_sites.csv' 

# Load intersection of sites in new dataset
site_info = pd.read_csv(site_info_path, index_col=0)
amdata = ad.read_h5ad(amdata_path, 'r')
params = modelling_bio.get_site_params()

intersection = site_info.index.intersection(amdata.obs.index)
amdata = amdata[intersection].to_memory()

# Remove values outside out beta distribution
amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)

amdata = amdata.copy()

# Add ground truth from wave3
amdata.obs[params + ['r2']] = site_info[params + ['r2']]

# Infer the offsets
maps = modelling_bio.site_offsets(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata.obs['offset'] = maps['offset']
sns.histplot(amdata.obs.offset)
amdata = amdata[amdata.obs.sort_values('offset').index]

# Apply the offset and refit acceleration and bias
offset = np.broadcast_to(amdata.obs.offset, shape=(amdata.shape[1], amdata.shape[0])).T
amdata.X = amdata.X - offset
ab_maps = modelling_bio.person_model(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']

amdata.var['acc'] = ab_maps['acc']
amdata.var['bias'] = ab_maps['bias']

sns.jointplot(amdata.var, x='acc', y='bias')

amdata.write_h5ad('../exports/wave1_acc.h5ad')
amdata.obs.to_csv('../exports/wave1_acc_sites.csv')
amdata.obs.to_csv('../exports/wave1_participants.csv')