'''
Apply the model on a external dataset
'''

# %%
# IMPORTS
# %load_ext autoreload
# %autoreload 2

import sys
import getopt
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

from src import modelling_bio_beta as model

N_CORES = 15

if len(sys.argv)==3:
    # python external.py wave1 wave3
    EXT_DSET_NAME = sys.argv[1] # external dataset name
    REF_DSET_NAME = sys.argv[2] # reference datset name
else:
    EXT_DSET_NAME = 'wave1' # external dataset name
    REF_DSET_NAME = 'wave3' # reference datset name


# %%
# LOAD

print(f'Running {REF_DSET_NAME} model on the {EXT_DSET_NAME} dataset')

# amdata = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{EXT_DSET_NAME}_fitted.h5ad', backed='r')
amdata = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{EXT_DSET_NAME}_meta.h5ad', backed='r')
participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/{EXT_DSET_NAME}_participants.csv', index_col='Basename')
amdata_ref = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{REF_DSET_NAME}_person_fitted.h5ad', backed='r')

# Load intersection of sites in new dataset
params = list(model.SITE_PARAMETERS.values())

# intersection = site_info.index.intersection(amdata.obs.index)
intersection = amdata_ref.obs.index.intersection(amdata.obs.index)

amdata = amdata[intersection].to_memory()
amdata.obs[params] = amdata_ref.obs[params]

# %% #################
# BATCH (MODEL) CORRECTION

amdata_chunks = model.make_chunks(amdata, chunk_size=30)

# if 'status' in amdata.var.columns:
#     maps = model.site_offsets(amdata[:,amdata.var.status=='control'], show_progress=True)
# else:
#     maps = model.site_offsets(amdata[:10], show_progress=True)

with Pool(N_CORES) as p:
    map_chunks = list(tqdm(p.imap(model.site_offsets, amdata_chunks), total=len(amdata_chunks)))

# maps = [model.site_offsets(chunk) for chunk in tqdm(amdata_chunks)]

# model.site_offsets(amdata_chunks[0])

offsets = np.concatenate([chunk['offset'] for chunk in map_chunks])


# # Infer the offsets
amdata.obs['offset'] = offsets
amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset
# sns.histplot(amdata.obs.offset, bins=50)
# amdata = amdata[amdata.obs.sort_values('offset').index].copy()

# %% #################
# show the offset applied to data

# site_index = amdata.obs.offset.abs().sort_values().index[-1]
# sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten(), label=EXT_DSET_NAME)
# sns.scatterplot(x=amdata_ref.var.age, y=amdata_ref[site_index].X.flatten(), label=REF_DSET_NAME)
# sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten()-amdata[site_index].obs.offset.values)
# %% ##################
# PERSON MODELLING  

ab_maps = model.person_model(amdata, method='map', progressbar=True)

participants[f'acc_{REF_DSET_NAME}'] = ab_maps['acc']
participants[f'bias_{REF_DSET_NAME}'] = ab_maps['bias']

# compute log likelihood for infered parameters to perform quality control
participants[f'qc_{REF_DSET_NAME}'] = model.get_fit_quality(amdata, 
                                         acc_name=f'acc_{REF_DSET_NAME}',
                                         bias_name=f'bias_{REF_DSET_NAME}')

participants.to_csv(f'{paths.DATA_PROCESSED_DIR}/{EXT_DSET_NAME}_participants.csv')
# %%

