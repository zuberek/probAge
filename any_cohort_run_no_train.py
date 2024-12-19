"""
ProbAge Inference Script
=======================

This script performs epigenetic age acceleration inference on methylation data using the ProbAge model.
It handles data loading, batch correction, and inference of acceleration and bias parameters for each participant.

Required input files:
    - Methylation data CSV (samples as columns, CpG sites as rows)
    - Metadata CSV (sample information including age and control status)
        - one can use the whole cohort as control status if there are no group distinction in the samples.
    - Pre-trained model site information (provided in streamlit/wave3_sites.csv)

Output files:
    - H5AD file containing both participant and CpG-level information
    - CSV file with updated metadata including
        - acceleration and bias parameters
        - log-likelihood and quality control measures
"""

from src.general_imports import *
from src import modelling_bio_beta as modelling
from src import batch_correction as bc
from sklearn.feature_selection import r_regression


###### GLOBAL PARAMETERS TO SET
# Set paths for loading and dumping data
path_to_data = 'data/wave4_data.csv'
path_to_metadata = 'data/wave4_metadata.csv'
h5ad_export_path = 'data/fitted_cohort.h5ad'
pandas_export_path = 'data/person_fit.csv'

# set number of cores used for inference (as low as 1)
MULTIPROCESSING = True
N_CORES = 32

# set wether we want to use beta or normal function for model fit
beta_fit = True

###### START OF INFERENCE
# Load data into anndata format
data_df = pd.read_csv(path_to_data, index_col=0)
metadata_df = pd.read_csv(path_to_metadata, index_col=0)
metadata_df['control'] = np.random.choice([True, False], size=len(metadata_df))  

# Create Anndata making sure samples are overlapping
sample_list = set(data_df.columns)
adata = ad.AnnData(data_df[list(sample_list)],
                    var=metadata_df.loc[list(sample_list)])

# fill site information from pre-trained model
sites_ref = pd.read_csv('streamlit/wave3_sites.csv', index_col=0)
adata = adata[adata.obs.index.isin(sites_ref.index)].copy()
adata.obs = sites_ref.loc[list(adata.obs.index)]

#
# # Load intersection of sites in new dataset
params = list(modelling.SITE_PARAMETERS.values())
print('Inferring site offsets for batch correction...')
offsets = bc.site_offsets(adata[:, adata.var.control == True], show_progress=True)['offset']
# print('Inferring site offsets ✅')

adata.obs['offset'] = offsets.astype('float64')
adata.obs.eta_0 = adata.obs.eta_0 + adata.obs.offset
adata.obs.meth_init  = adata.obs.meth_init + adata.obs.offset

# check that sites are properly fitted
site_index = -1
modelling.bio_model_plot(adata[site_index])

print('Inferring participants accelerations and biases...  ')

adata_chunks = modelling.make_chunks(adata.T, chunk_size=10)
adata_chunks = [chunk.T for chunk in adata_chunks]

partial_func = partial(modelling.person_model,
                       method='map',
                       progressbar=False,
                       map_method='L-BFGS-B', beta=beta_fit)

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

# export h5ad file with both participant and CpG-level information
adata.write_h5ad(h5ad_export_path)

# export updated metadata file
adata.var.to_csv(pandas_export_path)