# %% ########################
### LOADING

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as model

N_CORES = 15
N_SITES = 512 # number of sites to take in for the final person inferring

MULTIPROCESSING = True
DATASET_NAME = 'wave4'

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_fitted.h5ad')

# site_indexes = amdata[~amdata.obs.saturating].obs.sort_values('spr2').tail(100).index
# axs = plot.tab(site_indexes, ncols=10)
# for i, ax in enumerate(axs):
#     modelling_bio.bio_model_plot(amdata[site_indexes[i]], ax=ax)

# %% ########################
### LOADING

# select only not saturating sites
print(f'There are {(~amdata.obs.saturating==1).sum()} not saturating sites in {DATASET_NAME}')
amdata = amdata[~amdata.obs.saturating]

# further select sites
site_indexes = amdata.obs.sort_values('spr2').tail(N_SITES).index
amdata = amdata[site_indexes].to_memory()

# %% ########################
### MODELLING PEOPLE

amdata_chunks = model.make_chunks(amdata.T, chunk_size=15)
amdata_chunks = [chunk.T for chunk in amdata_chunks]

if MULTIPROCESSING:
    with Pool(N_CORES) as p:
        map_chunks = list(tqdm(p.imap(model.person_model,
                                      amdata_chunks
                                     )
                                ,total=len(amdata_chunks)
                                )
                            )

if not MULTIPROCESSING:
    map_chunks = map(model.person_model, amdata_chunks)

# %% ########################
### SAVING

for param in ['acc', 'bias']:
    param_data = np.concatenate([map[param] for map in map_chunks])
    amdata.var[param] = param_data

amdata.write_h5ad(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_person_fitted.h5ad')
# %%
