# %% ########################
# IMPORTING
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

import nutpie
import pymc as pm
import pickle
pm.model.make_initial_point_fn = pm.initial_point.make_initial_point_fn
from src import modelling_bio

dataset_name = 'ewas'

# %% ########################
# LOADING
params = modelling_bio.get_site_params()

chunk_size = 100
i=0
amdata_chunk1 = amdata_src.AnnMethylData(f'../resources/wave4_{i*chunk_size}to{(i+1)*chunk_size}_fitted.h5ad')
amdata = amdata_chunk1
for i in range(1,3000//chunk_size):
    amdata_chunk = amdata_src.AnnMethylData(f'../resources/wave4_{i*chunk_size}to{(i+1)*chunk_size}_fitted.h5ad')
    amdata = ad.concat([amdata, amdata_chunk])

amdata.obs.spr2.hist()
amdata_chunk2 = amdata_src.AnnMethylData(f'../resources/wave4_{i*chunk_size}to{(i+1)*chunk_size}_fitted.h5ad')



amdata.var = amdata_chunk1.var
modelling_bio.bio_model_plot(amdata['cg09670263'])
amdata.obs.spr2.max()


az.plot_trace(traces[3], var_names=params)
az.summary(traces[3], var_names=params)
amdata_chunk[3].obs
amdata = amdata_src.AnnMethylData('../exports/ewasKNN_fitted.h5ad', backed='r')
amdata = amdata[amdata.obs.sort_values('spr2').tail(n_sites).index]
amdata = amdata.to_memory()
scp disk/NSEA/exports/ewasKNN_fitted.h5ad jan@tina:~/tchandra-lab/Jan/methylation_data .

