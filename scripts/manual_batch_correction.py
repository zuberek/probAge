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

from src import modelling_bio_beta as modelling

N_CORES = 15
REF_DSET_NAME = 'probage_bc'
EXPORT_FILE_PATH = '../exports/processed_file.csv'

# Set paths to external data
path_to_data = '../data/downsyndrome.csv'
path_to_meta = '../data/downsyndrome_meta.csv'

# Load external datasets to pandas
data_df = pd.read_csv(path_to_data, index_col=0)
meta_df = pd.read_csv(path_to_meta, index_col=0)

# intersection of indexes between data and meta
part_index_intersection = data_df.columns.intersection(meta_df.index)

# create anndata object
amdata = ad.AnnData(data_df[part_index_intersection],
                    var=meta_df.loc[part_index_intersection])


# Load reference sites
sites_ref = pd.read_csv('../resources/wave3_sites.csv', index_col=0)
# amdata = ad.read_h5ad('resources/downsyndrome.h5ad')

# Load intersection of sites in new dataset
params = list(modelling.SITE_PARAMETERS.values())

intersection = sites_ref.index.intersection(amdata.obs.index)
amdata.obs[params] = sites_ref[params]


amdata = amdata[intersection].copy()

# %% #################
# BATCH (MODEL) CORRECTION

# Create amdata chunks using only control
amdata_chunks = modelling.make_chunks(amdata[:, amdata.var.status=='control'],
                                      chunk_size=15)

# print('Calculating the offsets...')
# if 'status' in amdata.var.columns:
#     offsets_chunks = [model.site_offsets(chunk[:,amdata.var.status=='healthy']) for chunk in tqdm(amdata_chunks)]
# else:
#     offsets_chunks = [model.site_offsets(chunk) for chunk in tqdm(amdata_chunks)]

with Pool(N_CORES) as p:
    offsets_chunks = list(tqdm(p.imap(modelling.site_offsets, amdata_chunks), total=len(amdata_chunks)))


offsets = np.concatenate([chunk['offset'] for chunk in offsets_chunks])

# # Infer the offsets
amdata.obs['offset'] = offsets
amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset

# %% ##################
# PERSON MODELLING  
print('Calculating person parameters (acceleration and bias)...')

# ab_maps = model.person_model(amdata, method='map', progressbar=True, map_method=None)

amdata_chunks = modelling.make_chunks(amdata.T, chunk_size=15)
amdata_chunks = [chunk.T for chunk in amdata_chunks]
with Pool(N_CORES) as p:
    map_chunks = list(tqdm(p.imap(modelling.person_model, amdata_chunks)
                            ,total=len(amdata_chunks)))

for param in ['acc', 'bias']:
    param_data = np.concatenate([map[param] for map in map_chunks])
    amdata.var[f'{param}_{REF_DSET_NAME}'] = param_data

# Export
amdata.var.to_csv(EXPORT_FILE_PATH)
