# %% ########################
# IMPORTING
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import modelling_bio_beta as model

DATASET_NAME = 'wave3'
N_PART = 500

n_sites_grid = [2**n for n in range(1,11)]
n_sites_label = [f'{n_sites_grid[i]}-{n_sites_grid[i+1]}' for i in range(len(n_sites_grid))[:-1]]


# %% ########################
# LOADING
amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_sites_fitted.h5ad',backed='r')
amdata = amdata[~amdata.obs.saturating].to_memory()
# Sort sites by spr2
amdata = amdata[amdata.obs.sort_values('spr2', ascending=False).index]

# select participants
part_indexes = model.sample_to_uniform_age(amdata, N_PART)
amdata = amdata[:, part_indexes].copy()


# %% ########################
# DOWNSAMPLING
accs = np.empty((len(n_sites_grid), amdata.shape[1]))
biases = np.empty((len(n_sites_grid), amdata.shape[1]))
for i, n_sites in enumerate(tqdm(n_sites_grid)):
    map = model.person_model(amdata[:n_sites], method='map' )
    accs[i] = map['acc']
    biases[i] = map['bias']    
accs = pd.DataFrame(accs, index=n_sites_grid, columns=amdata.var.index).T
biases = pd.DataFrame(biases, index=n_sites_grid, columns=amdata.var.index).T

# %%
accs.to_csv(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_accs.csv')
biases.to_csv(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_biases.csv')

