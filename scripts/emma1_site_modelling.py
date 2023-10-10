# %% ########################
# IMPORTING

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

N_CORES = cpu_count()


from src import modelling_bio

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

# %% ########################
# LOADING
amdata = amdata_src.AnnMethylData('../exports/wave4_meta.h5ad', backed='r')
amdata = amdata[amdata.obs.sort_values('spr2').tail(10).index]
amdata = amdata.to_memory()

chunk_size = 2
n_sites = amdata.shape[0]
amdata_chunks = []
for i in range(0, n_sites, chunk_size):
    amdata_chunks.append(amdata[i:i+chunk_size])

# %% ########################
# SITE MODELLING
with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=partial(modelling_bio.bio_sites, 
                            method='nuts', 
                            show_progress=True),
                iterable=amdata_chunks,
                chunksize=1
                ), 
            total=len(amdata_chunks)))

params = modelling_bio.get_site_params()
processed_results = []
for trace_bio in tqdm(results):    
    bio_fit = az.summary(trace_bio, var_names=params, round_to=5)
    bio_fit.index = pd.MultiIndex.from_tuples([(index_tuple[1][:-1], 'bio', index_tuple[0]) for index_tuple in bio_fit.index.str.split('[')],
                            names=['site', 'model', 'param'])
    processed_results.append(bio_fit)

fits = pd.DataFrame()
for fit in processed_results:
        fits = pd.concat([fits, fit])
# Save elpd_loo
# Extract parameter names from bio model
fits.to_csv('wave4_1k_ADVI_fits.csv')

param_list = fit.xs('bio', level='model').index.get_level_values(level='param')
for param in params:
    amdata.obs[f'{param}'] = fits.loc[(slice(None),'bio', param)]['mean'].values

amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(amdata, t=90)
amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(amdata)
amdata.obs['saturating_der'] = amdata.obs.abs_der<0.0005

amdata.obs['saturating'] = amdata.obs.saturating_std | amdata.obs.saturating_der

# Save just in case it breaks later
amdata.write_h5ad('../exports/wave4_fitted.h5ad')

# %% ########################
# PERSON MODELLING

amdata_clean = amdata[~amdata.obs.saturating]
amdata_clean = amdata_clean[amdata_clean.obs.sort_values('r2').tail(500).index].copy()

with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=partial(modelling_bio.person_model, 
                            method='nuts', 
                            show_progress=False),
                iterable=[amdata_clean],
                chunksize=1
                ), 
            total=amdata_clean.shape[1]))
    
processed_results = []
for trace_bio in tqdm(results):    
    bio_fit = az.summary(trace_bio, round_to=5)
    bio_fit.index = pd.MultiIndex.from_tuples([(index_tuple[1][:-1], 'bio', index_tuple[0]) for index_tuple in bio_fit.index.str.split('[')],
                            names=['site', 'model', 'param'])
    processed_results.append(bio_fit)

fits = pd.DataFrame()
for fit in processed_results:
        fits = pd.concat([fits, fit])

param_list = fit.xs('bio', level='model').index.get_level_values(level='param')
for param in param_list:
    amdata.var[f'{param}'] = fits.loc[(slice(None),'bio', param)]['mean'].values

amdata.write_h5ad('../exports/wave4_fitted.h5ad')