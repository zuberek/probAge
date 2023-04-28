# %%
# IMPORT
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling
from src import modelling_bio
import arviz as az

import pickle

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

N_SITES = slice(None)
N_PARTS = slice(None)
N_CORES = 15

OPEN_PATH = '../exports/ewas_linear.h5ad'
SAVE_PATH = '../exports/ewas_model_comparison.h5ad'

# %%
# LOAD

amdata = amdata_src.AnnMethylData(OPEN_PATH)
# amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index[N_SITES]].to_memory()
# amdata = amdata_src.AnnMethylData(amdata)
amdata.X = np.where(amdata.X == 0 , 0.0001, amdata.X)

# %%
# MODEL COMPARISON

chunks = modelling.chunkify_sites(amdata, N_CORES=100)

with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=partial(modelling.fit_and_compare, show_progress=False),
                iterable=chunks,
                ), 
            total=len(chunks)))
# %%
fits, comparisons = [], []
for site_index in tqdm(amdata[:10].sites.index):
    fit, comparison = modelling.fit_and_compare(amdata[site_index], cores=4, show_progress=False)
    fits.append(fit)
    comparisons.append(comparison)

# %%
fits, comparisons = modelling_bio.comparison_postprocess(results, amdata)
fits.to_csv('fits.csv')
comparisons('comparisons.csv')
# amdata.write_h5ad(SAVE_PATH)