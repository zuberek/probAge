'''
Apply the model on a external dataset
'''

# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

from src import modelling_bio_beta as model

# %%
# LOAD
EXTERNAL_DATASET_NAME = 'wave1'
REFERENCE_DATASET_NAME = 'ewasKNN'


# amdata = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{EXTERNAL_DATASET_NAME}_fitted.h5ad', backed='r')
amdata = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{EXTERNAL_DATASET_NAME}_meta.h5ad', backed='r')
participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/{EXTERNAL_DATASET_NAME}_participants.csv', index_col='Basename')
amdata_ref = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{REFERENCE_DATASET_NAME}_person_fitted.h5ad', backed='r')

# Load intersection of sites in new dataset
params = list(model.SITE_PARAMETERS.values())

# intersection = site_info.index.intersection(amdata.obs.index)
intersection = amdata_ref.obs.index.intersection(amdata.obs.index)

amdata = amdata[intersection].to_memory()
amdata.obs[params + ['r2', 'spr2']] = amdata_ref.obs[params + ['r2', 'spr2']]

# %% #################
# BATCH (MODEL) CORRECTION
if 'status' in amdata.var.columns:
    maps = model.site_offsets(amdata[:,amdata.var.status=='control'], show_progress=True)
else:
    maps = model.site_offsets(amdata, show_progress=True)

# Infer the offsets
amdata.obs['offset'] = maps['offset']
sns.histplot(amdata.obs.offset, bins=50)
# amdata = amdata[amdata.obs.sort_values('offset').index].copy()

# %% #################
# BATCH (MODEL) CORRECTION
# apply the offsets
# amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
# amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset

# show the offset applied to data
# site_index = amdata.obs.offset.abs().sort_values().index[-1]
# site_index = amdata.obs.spr2.abs().sort_values().index[-10]
sns.scatterplot(x=amdata.var.age, y=amdata['cg02503376'].X.flatten(), label=EXTERNAL_DATASET_NAME)
sns.scatterplot(x=amdata_ref.var.age, y=amdata_ref['cg02503376'].X.flatten(), label=REFERENCE_DATASET_NAME)

# sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten()-amdata[site_index].obs.offset.values)
# %% ##################
# PERSON MODELLING  

ab_maps = model.person_model(amdata, method='map', progressbar=True)

participants[f'acc_{REFERENCE_DATASET_NAME}'] = ab_maps['acc']
participants[f'bias_{REFERENCE_DATASET_NAME}'] = ab_maps['bias']

participants.to_csv(f'{paths.DATA_PROCESSED_DIR}/{EXTERNAL_DATASET_NAME}_participants.csv')
# %%
