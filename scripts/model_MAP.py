%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

# set to None to select all sites/participants
N_SITES = 4096
N_PARTS = None
N_CORES = 10
SITES, PARTICIPANTS = slice(N_SITES), slice(N_PARTS)


DATA_PATH = '../exports/wave3_meta.h5ad'


amdata = amdata_src.AnnMethylData(DATA_PATH, backed='r')
# amdata = amdata[amdata.obs.sort_values('r2', ascending=False)[SITES].index].to_memory()
amdata = amdata[amdata[amdata.obs.r2>0.2].obs.index].to_memory()
amdata = amdata_src.AnnMethylData(amdata)

#####################################
### MODELLING SITES

chunk_size = amdata.shape[0]//N_CORES
n_sites = amdata.shape[0]
amdata_chunks = []
for i in range(0, n_sites, chunk_size):
    amdata_chunks.append(amdata[i:i+chunk_size])

with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=partial(modelling_bio.bio_sites, 
                            return_MAP=True, 
                            return_trace=False, 
                            show_progress=True),
                iterable=amdata_chunks,
                chunksize=1
                ), 
            total=len(amdata_chunks)))

site_maps = modelling_bio.concat_maps(results)
for param in modelling_bio.get_site_params():
    amdata.obs[param] = site_maps[param].values

amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(amdata, t=90)
amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(amdata)
amdata.obs['saturating_der'] = amdata.obs.abs_der<0.001

amdata.obs['saturating'] = amdata.obs.saturating_std | amdata.obs.saturating_der

amdata.write_h5ad('../exports/wave3_all_fitted.h5ad')
amdata = ad.read_h5ad('../exports/wave3_all_fitted.h5ad')

#####################################
### MODELLING PEOPLE

# chunk_size = amdata.shape[1]//N_CORES
# n_participants = amdata.shape[1]
# amdata_chunks = []
# for i in range(0, n_participants, chunk_size):
#     amdata_chunks.append(amdata[:,i:i+chunk_size])

# with Pool(N_CORES, maxtasksperchild=1) as p:
#     results = list(tqdm(
#             iterable= p.imap(
#                 func=partial(modelling_bio.person_model, 
#                             return_MAP=True, 
#                             return_trace=False, 
#                             show_progress=True),
#                 iterable=amdata_chunks,
#                 chunksize=1
#                 ), 
#             total=len(amdata_chunks)))
# person_maps = modelling_bio.concat_maps(results)

amdata = amdata[~amdata.obs.saturating]
amdata = amdata[amdata.obs.sort_values('r2').tail(250).index].copy()

person_maps = modelling_bio.person_model(
                            amdata, 
                            return_trace=False, return_MAP=True, show_progress=True)['map']

amdata.var['acc'] = person_maps['acc']
amdata.var['bias'] = person_maps['bias']



amdata.write_h5ad('../exports/wave3_acc.h5ad')
amdata.obs.to_csv('../exports/wave3_acc_sites.csv')