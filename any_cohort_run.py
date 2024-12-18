 # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling
from src import batch_correction as bc

# adata = ad.read_h5ad('data/wave4_meta_3000.h5ad')

# Load data into anndata format
path_to_data = 'data/wave4_data.csv'
path_to_metadata = 'data/wave4_metadata.csv'

# set number of cores used for inference (as low as 1)
MULTIPROCESSING = True
N_CORES = 32

data_df = pd.read_csv(path_to_data, index_col=0)
metadata_df = pd.read_csv(path_to_metadata, index_col=0)

# Create Anndata making sure samples are overlapping
sample_list = set(data_df.columns)
adata = ad.AnnData(data_df[list(sample_list)],
                    var=metadata_df.loc[list(sample_list)])

# fill site information from pre-trained model
sites_ref = pd.read_csv('streamlit/wave3_sites.csv', index_col=0)
adata = adata[adata.obs.index.isin(sites_ref.index)].copy()
adata.obs = sites_ref.loc[list(adata.obs.index)]

# Load intersection of sites in new dataset
params = list(modelling.SITE_PARAMETERS.values())
print('Inferring site offsets... ')
offsets = bc.site_offsets(adata, show_progress=True)['offset']
# print('Inferring site offsets ✅')

    
adata.obs['offset'] = offsets.astype('float64')
adata.obs.eta_0 = adata.obs.eta_0 + adata.obs.offset
adata.obs.meth_init  = adata.obs.meth_init + adata.obs.offset

# check that sites are properly fitted
site_index = -1
modelling.bio_model_plot(adata[site_index])

print('Inferring participants accelerations and biases...  ')

adata = adata[:, :1_000].copy()
adata_chunks = modelling.make_chunks(adata.T, chunk_size=10)
adata_chunks = [chunk.T for chunk in adata_chunks]

partial_func = partial(modelling.person_model,
                       method='map',
                       progressbar=False,
                       map_method='L-BFGS-B', beta=True)

if MULTIPROCESSING:
    with Pool(N_CORES) as p:
        map_chunks = list(tqdm(p.imap(partial_func,
                                      adata_chunks
                                     )
                                ,total=len(adata_chunks)
                                )
                            )

if not MULTIPROCESSING:
    map_chunks = map(partial_func, adata_chunks)

### SAVING
for param in ['acc', 'bias']:
    param_data = np.concatenate([map[param] for map in map_chunks])
    adata.var[param] = param_data

# compute log likelihood for infered parameters to perform quality control
ab_ll = modelling.person_model_ll(adata)
adata.var['ll'] = ab_ll
adata.var['qc'] = modelling.get_person_fit_quality(ab_ll)


sns.scatterplot(adata.var, x='acc', y='bias', hue='ll')

adata.write_h5ad(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_person_fitted.h5ad')

metadata_df['acc'] = adata.var['acc']
metadata_df['bias'] = adata.var['bias']
metadata_df['ll'] = adata.var['ll']
metadata_df['qc'] = adata.var['qc']

metadata_df.to_csv(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_participants.csv')
