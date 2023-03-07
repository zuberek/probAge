# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
from src import preprocess_func

# %%
# LOAD

amdata = ad.read_h5ad('../exports/lbc.h5ad', backed='r')
amdata2 = ad.read_h5ad('../exports/wave3_MAP_acc.h5ad')
site_info_path = '../exports/wave3_acc_sites.csv' 

# Load intersection of sites in new dataset
site_info = pd.read_csv(site_info_path, index_col=0)
params = modelling_bio.get_site_params()

intersection = site_info.index.intersection(amdata.obs.index)
amdata = amdata[intersection].to_memory()

amdata = amdata[:,amdata.var.reset_index().set_index('ID').loc[amdata.var.ID.value_counts()[amdata.var.ID.value_counts()>2].index].set_index('index').index]
amdata = amdata[:, amdata.var.cohort=='LBC36']


amdata.obs[params + ['r2']] = site_info[params + ['r2']]
amdata = amdata.copy()

# %%

amdata2[0].X.mean()
amdata[0].X.max()


# Infer the offsets
maps = modelling_bio.site_offsets(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata.obs['offset'] = maps['offset']
sns.histplot(amdata.obs.offset)
amdata = amdata[amdata.obs.sort_values('offset').index]
