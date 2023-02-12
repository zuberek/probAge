# %load_ext autoreload 
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)
import pickle

# set to None to select all sites/participants
N_SITES = 4096
N_SITES = 15
N_PARTS = None
N_CORES = 10
SITES, PARTICIPANTS = slice(N_SITES), slice(N_PARTS)


DATA_PATH = '../exports/wave3_meta.h5ad'


amdata = amdata_src.AnnMethylData(DATA_PATH, backed='r')
# amdata = amdata[amdata.obs.sort_values('r2', ascending=False)[SITES].index].to_memory()
amdata = amdata[amdata[amdata.obs.r2>0.2].obs.index].to_memory()
amdata = amdata_src.AnnMethylData(amdata)

#####################################
### MODELLING SITES
amdata = amdata_src.AnnMethylData('../exports/wave3_acc.h5ad')
amdata.obs
modelling_bio.bio_model_plot(amdata[-2])
amdata[0].obs
res = modelling_bio.bio_sites(amdata[-2], show_progress=True, cores=4)['trace']
res1 = modelling_bio.bio_sites(amdata[1], show_progress=True, cores=4)['trace']
res2 = modelling_bio.bio_sites(amdata[2], show_progress=True, cores=4)['trace']
res = res['trace']

# We can plot the posterior for each individual assigning coordinates
import arviz as az
part_idx = filtered_sites.obs.index[0]
az.plot_pair(res,kind=['hexbin', 'kde'], point_estimate='mode', marginals=True)
az.plot_posterior(res)

chunk_size = amdata.shape[0]//(N_CORES*2)
chunk_size = 1
n_sites = amdata.shape[0]
amdata_chunks = []
for i in range(0, n_sites, chunk_size):
    amdata_chunks.append(amdata[i:i+chunk_size])

with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=partial(modelling_bio.fit_and_compare, 
                            show_progress=True
                            ),
                iterable=amdata_chunks,
                chunksize=1
                ), 
            total=len(amdata_chunks)))

fits, comparisons = modelling_bio.comparison_postprocess(results, amdata)
comparisons.to_csv('wave3_250_comparisons.csv')
fits.to_csv('wave3_250_fits.csv')
fits.r_hat.hist()
comparisons.warning.hist()

with open('../exports/results.pk', 'wb') as f:
    pickle.dump(results, f)

map_sites = pd.read_csv('../exports/wave3_sites.csv', index_col=0)
sns.scatterplot(x=map_sites.omega, y=amdata.obs.omega)



#%%

# site_maps = modelling_bio.concat_maps(results)
for param in modelling_bio.get_site_params():
    amdata.obs[f'map_{param}'] = map_sites[param]

# amdata.obs['saturating'] = modelling_bio.is_saturating_vect(amdata)

# sns.scatterplot(data=amdata.obs, x='system_size', y='var_init', hue='saturating')
# sns.scatterplot(data=amdata.obs, x='r2', y='var_init', hue='saturating')
# sns.histplot(amdata.obs.var_init, log_scale=True)
# sns.histplot(data=amdata.obs, x='r2', hue='saturating')

# amdata = amdata[~amdata.obs.saturating].copy()

# #####################################
# ### MODELLING PEOPLE

# # chunk_size = amdata.shape[1]//N_CORES
# # n_participants = amdata.shape[1]
# # amdata_chunks = []
# # for i in range(0, n_participants, chunk_size):
# #     amdata_chunks.append(amdata[:,i:i+chunk_size])

# # with Pool(N_CORES, maxtasksperchild=1) as p:
# #     results = list(tqdm(
# #             iterable= p.imap(
# #                 func=partial(modelling_bio.person_model, 
# #                             return_MAP=True, 
# #                             return_trace=False, 
# #                             show_progress=True),
# #                 iterable=amdata_chunks,
# #                 chunksize=1
# #                 ), 
# #             total=len(amdata_chunks)))
# # person_maps = modelling_bio.concat_maps(results)

# person_maps = modelling_bio.person_model(amdata, return_trace=False, 
#                                     return_MAP=True, show_progress=True)['map']

# amdata.var['acc'] = person_maps['acc']
# amdata.var['bias'] = person_maps['bias']


# sns.jointplot(amdata.var, x='acc', y='bias')

# amdata.write_h5ad('../exports/wave3_MAP.h5ad')