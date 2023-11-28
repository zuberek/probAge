# %% ########################
### LOADING

# %load_ext autoreload 
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as model

N_CORES = 15
N_SITES = 3_000 # number of sites to take in for the final person inferring
N_PART = 2_000

MULTIPROCESSING = True

if len(sys.argv)==2:
    # python external.py wave1 wave3
    DSET_NAME = sys.argv[1] # external dataset name
else:
    DSET_NAME = 'wave3' # external dataset name

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DSET_NAME}_meta.h5ad', backed='r')

# select sites
site_indexes = amdata.obs.sort_values('spr2').tail(N_SITES).index
amdata = amdata[site_indexes].to_memory()

# select participants
amdata = amdata[:,amdata.var.age<60]
part_indexes = model.sample_to_uniform_age(amdata, N_PART)
amdata = amdata[:, part_indexes].copy()
amdata.var.age.hist(bins=50)

print(f'Modelling sites for {DSET_NAME} dataset with {amdata.shape[0]} sites and {amdata.shape[1]} participants after selection...')

# %% ########################
### MODELLING SITES

amdata_chunks = model.make_chunks(amdata, chunk_size=15)

# single process
if not MULTIPROCESSING:
    map_chunks = [model.parallel_site_MAP(chunk) for chunk in tqdm(amdata_chunks)]

# multiprocessing
if MULTIPROCESSING:
    with Pool(N_CORES) as p:
        map_chunks = list(tqdm(p.imap(model.parallel_site_MAP, amdata_chunks), total=len(amdata_chunks)))



# %% ########################
### SAVING

# reload the dataset to keep all the participants
amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DSET_NAME}_meta.h5ad', backed='r')
amdata = amdata[site_indexes].to_memory()

# store results
for param in model.SITE_PARAMETERS.values():
    param_data = np.concatenate([map[param] for map in map_chunks])
    amdata.obs[param] = param_data
amdata = model.get_saturation_inplace(amdata)

# save
amdata.write_h5ad(f'{paths.DATA_PROCESSED_DIR}/{DSET_NAME}_sites_fitted_YOUNG.h5ad')