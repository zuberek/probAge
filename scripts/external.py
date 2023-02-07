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

external_path = '../exports/hannum.h5ad'
external_cohort_name = 'hannum'

N_SITES = False
N_PARTS = False
N_CORES = 7

# load selected sites for fitting
with open("../resources/selected_sites.json", "r") as f:
    sites = json.load(f)

# load external cohort
amdata = amdata_src.AnnMethylData(external_path, backed='r')
# filter sites
amdata = amdata[amdata.sites.index.isin(sites)].to_memory()
amdata = amdata_src.AnnMethylData(amdata)

if N_SITES is not False:
    amdata = amdata[:N_SITES]

with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling_bio.bio_fit,
                iterable=amdata,
                chunksize=1
                ), 
            total=amdata.n_obs))

fits = modelling_bio.bio_fit_post(results, amdata)

# Fitting acceleration and bias
print('Acceleration and bias model fitting')
if N_PARTS is not False:
    amdata = amdata[:, :N_PARTS]

# create amdata chunks to vectorize acc and bias over participants
chunk_size = 10
n_participants = amdata.shape[1]
amdata_chunks = []
for i in range(0, n_participants, chunk_size):
    amdata_chunks.append(amdata[:,i:i+chunk_size])

# Modify fitting function to return only MAP
person_model = partial(modelling_bio.person_model, return_MAP=True, return_trace=False)

results = []
for chunk in tqdm(amdata_chunks):
    results.append(
        person_model(chunk
        )
    )

# Append acc and bias to amdata object
acc = np.concatenate([r['map']['acc'] for r in results])
bias = np.concatenate([r['map']['bias'] for r in results])

amdata.var['acc_mean'] = acc
amdata.var['bias_mean'] = bias

# Export amdata object
amdata.write_h5ad('../exports/' + external_cohort_name +'_acc_bio.h5ad')