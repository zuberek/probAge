# %% ########################
### LOADING

%load_ext autoreload 
%autoreload 2

import scipy.stats as stats

import sys
sys.path.append("..")   # fix to import modules from root

from src.general_imports import *
from src import preprocess_func

N_CORES = 15

MULTIPROCESSING = True
DATASET_NAME = 'wave3'

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_meta.h5ad')

# %% ########################
### PREPROCESS

amdata_chunks = amdata.chunkify(chunksize=1_000)

for chunk in tqdm(amdata_chunks):
    chunk = preprocess_func.drop_nans(chunk)
    chunk.X = np.where(chunk.X == 0, 0.00001, chunk.X)
    chunk.X = np.where(chunk.X == 1, 0.99999, chunk.X)

# %% ########################
### CALCULATE SPEARMAN

def spearman_r_loop(site_idx):
    spr = stats.spearmanr(amdata[site_idx].X.flatten(), amdata.var.age)
    return spr.statistic

with Pool(N_CORES) as pool:
    result = list(tqdm(pool.imap(spearman_r_loop, amdata.obs.index), total=amdata.shape[0]))

amdata.obs['spr'] = result
amdata.obs['spr2'] = amdata.obs.spr**2

# %% ########################
### SAVE DATA
amdata = amdata[amdata.obs.sort_values('spr2').index]
amdata.write_h5ad(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_meta.h5ad')
# %%
