%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import modelling_bio

N_SITES = 20

site_indexes = amdata.obs.sort_values('r2').tail(N_SITES).index


axs = plot.tab(site_indexes, ncols=5, row_size=5)
for i, site_index in enumerate(site_indexes): 
    modelling_bio.bio_model_plot(amdata[site_index], ax=axs[i])