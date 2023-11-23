'''
Apply the model on a external dataset
'''

# %%
# IMPORTS
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

from src import modelling_bio_beta as model

N_CORES = 15

N_SITES = None
N_PARTICIPANTS = None

if len(sys.argv)==3:
    # python external.py wave1 wave3
    EXT_DSET_NAME = sys.argv[1] # external dataset name
    REF_DSET_NAME = sys.argv[2] # reference datset name
else:
    EXT_DSET_NAME = 'wave3' # external dataset name
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
intersection = amdata_ref.obs.index.intersection(amdata.obs.index)[:N_SITES]

amdata = amdata[intersection].to_memory()
amdata.obs[params] = amdata_ref.obs[params]

# amdata = amdata[:, amdata.var.age>18].copy()
# participants = participants[participants.age>18]



# %% #################
# BATCH (MODEL) CORRECTION

amdata_chunks = model.make_chunks(amdata, chunk_size=10)

# print('Calculating the offsets...')
# if 'status' in amdata.var.columns:
#     offsets_chunks = [model.site_offsets(chunk[:,amdata.var.status=='healthy']) for chunk in tqdm(amdata_chunks)]
# else:
#     offsets_chunks = [model.site_offsets(chunk) for chunk in tqdm(amdata_chunks)]

with Pool(N_CORES) as p:
    offsets_chunks = list(tqdm(p.imap(model.site_offsets, amdata_chunks), total=len(amdata_chunks)))
                     

offsets = np.concatenate([chunk['offset'] for chunk in offsets_chunks])

# # Infer the offsets
amdata.obs['offset'] = offsets
amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset
# sns.histplot(amdata.obs.offset, bins=100)

# %% #################
# show the offset applied to data

# site_index = amdata.obs.offset.abs().sort_values().index[-1]
# sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten(), label=EXT_DSET_NAME)
# sns.scatterplot(x=amdata_ref.var.age, y=amdata_ref[site_index].X.flatten(), label=REF_DSET_NAME)
# sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten()-amdata[site_index].obs.offset.values)

# %% ##################
# PERSON MODELLING  
print('Calculating person parameters (acceleration and bias)...')

# ab_maps = model.person_model(amdata, method='map', progressbar=True, map_method=None)

amdata_chunks = model.make_chunks(amdata.T, chunk_size=15)
amdata_chunks = [chunk.T for chunk in amdata_chunks]
with Pool(N_CORES) as p:
    map_chunks = list(tqdm(p.imap(model.person_model, amdata_chunks)
                            ,total=len(amdata_chunks)))
for param in ['acc', 'bias']:
    param_data = np.concatenate([map[param] for map in map_chunks])
    amdata.var[f'{param}_{REF_DSET_NAME}'] = param_data

# if ab_maps['acc'].sum() == 0:
#     print('Fitting error. Try different map method, for example Powell')

participants[f'acc_{REF_DSET_NAME}'] = amdata.var[f'acc_{REF_DSET_NAME}']
participants[f'bias_{REF_DSET_NAME}'] = amdata.var[f'bias_{REF_DSET_NAME}']

# amdata.var[f'acc_{REF_DSET_NAME}'] = ab_maps['acc']
# amdata.var[f'bias_{REF_DSET_NAME}'] = ab_maps['bias']
participants[f'll_{REF_DSET_NAME}'] = model.person_model_ll(amdata, 
                                                            acc_name=f'acc_{REF_DSET_NAME}', 
                                                            bias_name=f'bias_{REF_DSET_NAME}')

# compute log likelihood for infered parameters to perform quality control
participants[f'qc_{REF_DSET_NAME}'] = model.get_person_fit_quality(
    participants[f'll_{REF_DSET_NAME}'])

participants.to_csv(f'{paths.DATA_PROCESSED_DIR}/{EXT_DSET_NAME}_participants.csv')

# sns.scatterplot(x=participants.age, y=participants.bias_wave3, hue=participants.status)
# %%

# import plotly.express as px

# participants.acc_wave3.mean()
# participants['status_binary'] = (participants.status=='healthy')*1
# px.scatter(x=participants.status_binary, y=participants.acc_wave3, color=participants.sex, trendline='ols')
# px.scatter(x=participants.acc_wave3, y=participants.bias_wave3, color=participants.status, marginal_x='histogram' , marginal_y='histogram')