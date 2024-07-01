# %% ########################
### LOADING

# %load_ext autoreload 
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling

N_CORES = 15
N_SITES = 1024 # number of sites to take in for the final person inferring

MULTIPROCESSING = True
DATASET_NAME = 'wave3'

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_sites_fitted.h5ad')
amdata2 = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_sites_fitted.h5ad')
participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_participants.csv', index_col='Basename')

# site_indexes = amdata[~amdata.obs.saturating].obs.sort_values('spr2').tail(100).index
# axs = plot.tab(site_indexes, ncols=10)
# for i, ax in enumerate(axs):
#     modelling_bio.bio_model_plot(amdata[site_indexes[i]], ax=ax)

# %% ########################
### LOADING

# select only not saturating sites
print(f'There are {(~amdata.obs.saturating==1).sum()} not saturating sites in {DATASET_NAME}')
amdata = amdata[~amdata.obs.saturating]

# further select sites
site_indexes = amdata.obs.sort_values('spr2').tail(N_SITES).index
amdata = amdata[site_indexes].to_memory()

# %% ########################
### MODELLING PEOPLE

# ab_maps = model.person_model(amdata, method='map', progressbar=True)

# from pymc.variational.callbacks import CheckParametersConvergence
# import pymc as pm
# import arviz as az
# person_model = model.person_model(amdata[:,:10], method='map', progressbar=True)

# with person_model:
#     mean_field = pm.fit(method='advi', n=50_000, callbacks=[CheckParametersConvergence()],  progressbar=True)

# with person_model:
#     trace = pm.sample(progressbar=True)

# with person_model:
#     maps = pm.find_MAP(progressbar=True)



# plt.plot(mean_field.hist)
# trace = mean_field.sample(1_000)
# az.plot_trace(trace)


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

# %% ########################
# ### SAVING

for param in ['acc', 'bias']:
    param_data = np.concatenate([map[param] for map in map_chunks])
    amdata.var[param] = param_data

# compute log likelihood for infered parameters to perform quality control
ab_ll = modelling.person_model_ll(amdata)
amdata.var['ll'] = ab_ll
participants['qc'] = modelling.get_person_fit_quality(
    participants['ll'])


amdata.write_h5ad(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_person_fitted.h5ad')

participants['acc'] = amdata.var['acc']
participants['bias'] = amdata.var['bias']
participants['ll'] = amdata.var['ll']
participants['qc'] = amdata.var['qc']

participants.to_csv(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_participants.csv')
# %%

