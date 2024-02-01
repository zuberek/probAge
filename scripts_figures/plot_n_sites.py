%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import modelling_bio

N_SITES = 20


# top r2
site_indexes = amdata.obs.sort_values('r2').tail(N_SITES).index

site_indexes = amdata[~amdata.obs.saturating].obs.sort_values('r2', ascending=False).head(N_SITES).index
site_indexes = amdata.obs.sort_values('der', ascending=False).tail(N_SITES).index
site_indexes = amdata.obs[amdata.obs.saturating].sort_values('r2', ascending=False).head(N_SITES).index
site_indexes = amdata.obs.sort_values('var_init', ascending=False).head(N_SITES).index
site_indexes = amdata.obs[amdata.obs.var_init>1].sort_values('var_init', ascending=False).tail(N_SITES).index
site_indexes = amdata.obs[~amdata.obs.saturating * amdata.obs.saturating_old].tail(N_SITES).index
site_indexes = amdata.obs[amdata.obs.saturating].sort_values('r2', ascending=False).head(N_SITES).index

axs = plot.tab(site_indexes, ncols=5, row_size=5)
for i, site_index in enumerate(site_indexes): 
    modelling_bio.bio_model_plot(amdata[site_index], hue=amdata.var.status, ax=axs[i])