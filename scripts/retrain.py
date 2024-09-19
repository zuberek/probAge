# %% ########################
### LOADING

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling

# INPUT PATH TO DATA AND METADATA
DATA_PATH = '../data/downsyndrome.csv'
META_DATA_PATH = '../data/downsyndrome_meta.csv'

# PATH to OUTPUT
OUTPUT_PATH = '../exports/'

# Set the perfectage of participants used for training 
TRAINING_PERCENTAGE = 100

# Set multiprocessing option to parallelize computations or not
# and number of cores to be used
MULTIPROCESSING = True
N_CORES = 15

# load selected sites
with open('../resources/selected_sites.json') as f:
    selected_sites = json.load(f)

# load data and metadata
data = pd.read_csv(DATA_PATH, index_col=0)
m_data = pd.read_csv(META_DATA_PATH, index_col=0)

# create intersection between selected sites and sites included in data
selected_sites = (data.index).intersection(selected_sites)

# Create AnnData object
data = data.loc[selected_sites].copy()
data = data.sort_index(axis=1)
m_data = m_data.sort_index()

amdata = ad.AnnData(data, var=m_data)

# set number of participant used for training
N_PART = int(amdata.shape[1]*TRAINING_PERCENTAGE/100)

part_indexes = modelling.sample_to_uniform_age(amdata, N_PART)
amdata.var['training'] = False
amdata.var.loc[part_indexes, 'training'] = True

sns.histplot(amdata.var[amdata.var.training == True], x='age')
plt.title('Training set age distribution')
plt.show()
plt.clf()

# %% ########################
### MODELLING SITES
print(f'Modelling sites...')

# create site chunks for vectorization
# larger chunks might result in less accurate fits
amdata_training = amdata[:, amdata.var.training == True]
amdata_chunks = modelling.make_chunks(amdata_training, chunk_size=15)

# single process
if not MULTIPROCESSING:
    map_chunks = [modelling.site_MAP(chunk) for chunk in tqdm(amdata_chunks)]

# multiprocessing
if MULTIPROCESSING:
    with Pool(N_CORES) as p:
        map_chunks = list(tqdm(p.imap(modelling.site_MAP, amdata_chunks), total=len(amdata_chunks)))

# store results
for param in modelling.SITE_PARAMETERS.values():
    param_data = np.concatenate([map[param] for map in map_chunks])
    amdata.obs[param] = param_data

# We have now saved the results of modelling site dynamics 
# in the obs dataframe of amdata object.

# %% ########################
### MODELLING SITES
print(f'Modelling Acceleration and Bias...')


# create participant chunks for vectorization
# larger chunks might result in less accurate fits
amdata_chunks = modelling.make_chunks(amdata.T, chunk_size=15)
amdata_chunks = [chunk.T for chunk in amdata_chunks]

if MULTIPROCESSING:
    with Pool(N_CORES) as p:
        map_chunks = list(tqdm(p.imap(modelling.person_model,
                                      amdata_chunks
                                     )
                                ,total=len(amdata_chunks)
                                )
                            )

if not MULTIPROCESSING:
    map_chunks = map(modelling.person_model, amdata_chunks)

for param in ['acc', 'bias']:
    param_data = np.concatenate([map[param] for map in map_chunks])
    amdata.var[param] = param_data

# compute log likelihood for infered parameters to perform quality control
ab_ll = modelling.person_model_ll(amdata)
amdata.var['ll'] = ab_ll
amdata.var['qc'] = modelling.get_person_fit_quality(
    amdata.var['ll'])

sns.scatterplot(amdata.var, x='acc', y='bias', hue='qc')
plt.title('Overview of results qith QC')
plt.show()
plt.clf()

# %% ########################
### EXPORTING RESULTS
print('Exporting results')
# Export anndata object
amdata.write_h5ad(OUTPUT_PATH + f'probage_results.h5ad')
#export only participant information
amdata.var.to_csv(OUTPUT_PATH + f'probage_results_participant.csv')