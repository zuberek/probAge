%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import modelling_bio

params = modelling_bio.get_site_params()

for param in modelling_bio.get_site_params():
    amdata.obs[f'other_{param}'] = map_sites[param]


axs = plot.tab(params, 'Dataset comparison', ncols=2, row_size=5, col_size=8)
for i, param in enumerate(params):
    sns.scatterplot(x=amdata.obs[param], y=amdata.obs[f'other_{param}'], ax=axs[i])
