# %% ########################
### LOADING

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling

N_CORES = 15
N_SITES = 1024 # number of sites to take in for the final person inferring

DATASET_NAME = 'wave3'

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_person_fitted.h5ad')

# %% ########################
### PREP

amdata.var['acc_abs'] = np.abs(amdata.var.acc)
amdata = amdata[:, amdata.var.acc_abs.sort_values(ascending=False).index].copy()

subsets = [np.random.choice(amdata.var.index, replace=False, size=200) for i in range(20)]
subsets_maps = [modelling.site_MAP(amdata[:,subset]) for subset in tqdm(subsets)]

subsets_maps = []
for subset in tqdm(subsets):
    subsets_maps.append(modelling.site_MAP(amdata[:,subset]))
