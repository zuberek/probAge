# %% ########################
# IMPORTING
# scp /exports/igmm/eddie/tchandra-lab/EJYang/methylclock/ProbAge/exports/EWAS/*.h5ad .

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import modelling_bio
import nutpie
import pymc as pm
import pickle
pm.model.make_initial_point_fn = pm.initial_point.make_initial_point_fn

dataset = 'EWAS'

# %% ########################
# CREATE ANNDATA
i = 0
chunk_size = 100
n_sites = 3_000

amdata = amdata_src.AnnMethylData(f'../exports/{dataset}/{dataset}_{i*chunk_size}to{(i+1)*chunk_size}_fitted.h5ad')
var = amdata.var
for i in range(1,n_sites//chunk_size):
    amdata = ad.concat([amdata, amdata_src.AnnMethylData(f'../exports/{dataset}/{dataset}_{i*chunk_size}to{(i+1)*chunk_size}_fitted.h5ad')])
amdata.var = var
amdata = amdata[amdata.obs.sort_values('spr2',ascending=False).index]

# %% ########################
# SAVE ANNDATA
amdata.write_h5ad(f'../exports/EWAS_fitted.h5ad')
# %%
amdata.obs.spr2.hist()
modelling_bio.bio_model_plot(amdata[0])
# %%
axs = plot.tab(amdata.obs.index[:12], title=f'{dataset} top12')
for i, ax in enumerate(axs):
    modelling_bio.bio_model_plot(amdata[i], ax=ax)

amdata[:12].obs.spr2.hist()
# %%
