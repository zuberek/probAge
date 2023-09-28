# %% ########################
# IMPORTING
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths
from scipy import stats
import src.preprocess_func  as preprocess_func
from sklearn.feature_selection import r_regression

N_CORES = 15


from src import modelling_bio

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

# %% ########################
# DATA PREPROCESSSING
sample_meta = pd.read_csv('../../methylation/wave4/sample_meta.csv', index_col=0)
amdata = amdata_src.AnnMethylData('../../methylation/wave4/wave4.h5ad')
amdata.var[['age', 'sex', 'id']] = sample_meta[['age', 'sex', 'Sample_Name']]


chunk_size = 1_000
n_sites = amdata.shape[0]
amdata_chunks = []
for i in range(0, n_sites, chunk_size):
    amdata_chunks.append(amdata[i:i+chunk_size])

# Clean data
for chunk in tqdm(amdata_chunks):
    chunk = preprocess_func.drop_nans(chunk)
    chunk.X = preprocess_func.convert_mvalues_to_betavalues(chunk)
    chunk.X = np.where(chunk.X == 0, 0.00001, chunk.X)
    chunk.X = np.where(chunk.X == 1, 0.99999, chunk.X)

# Compute spearman
def spearman_r_loop(site_idx):
    spr = stats.spearmanr(amdata[site_idx].X.flatten(), amdata.var.age)
    return spr.statistic**2
with Pool(N_CORES) as p:
    result = list(tqdm(p.imap(spearman_r_loop, amdata.obs.index), total=amdata.shape[0]))
amdata.obs['spr2'] = result

sns.scatterplot(x=result, y=amdata.obs.spr2)

# Compute pearson
def r2(site_index):
    return r_regression(amdata[site_index].X.T, amdata.var.age)[0]**2
with Pool(N_CORES) as P:
    result = list(tqdm(P.imap(r2, amdata.obs.index), total=amdata.shape[0]))
amdata.obs['r2'] = result

amdata.write_h5ad(paths.DATA_PROCESSED)

# %% ########################
# SITE MODELLING

amdata = amdata_src.AnnMethylData(paths.DATA_PROCESSED, backed='r')
amdata = amdata[amdata.obs.sort_values('spr2').tail(100).index]
amdata = amdata.to_memory()

chunk_size = 2
n_sites = amdata.shape[0]
amdata_chunks = []
for i in range(0, n_sites, chunk_size):
    amdata_chunks.append(amdata[i:i+chunk_size])

amdata_site = amdata[1].copy()
import pymc as pm
advi_trace = modelling_bio.bio_sites(amdata_site, method='advi')


params = modelling_bio.get_site_params()
amdata_site.obs[params] = az.summary(advi_trace)['mean'].values
modelling_bio.bio_model_plot(amdata_site, alpha=0.1)

with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=partial(modelling_bio.bio_sites, 
                            method='nuts', 
                            show_progress=False),
                iterable=amdata_chunks,
                chunksize=1
                ), 
            total=len(amdata_chunks)))

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

# Plot comparison of NUTS ADVI MAP
# amdata.obs[[f'{param}_MAP' for param in params]].to_csv('ewas_100MAP.csv')

# param = 'eta_0'
# param_method = [f'{param}_MAP' for param in params] + [f'{param}_ADVI' for param in params] 

# axs = plot.tab(param_method, ncols=5, row_size=5)
# for i, param in enumerate(param_method):
#     sns.scatterplot(data=amdata.obs, x=param, y='_'.join(param.split('_')[:-1]), ax=axs[i])
#     sns.lineplot(x=(0, amdata.obs[param].max()), y=(0, amdata.obs[param].max()), ax=axs[i])



site_maps = pd.concat([pd.DataFrame(result) for result in results])
for param in modelling_bio.get_site_params():
    amdata.obs[f'{param}_MAP'] = site_maps[param].values

amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(amdata, t=90)
amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(amdata)
amdata.obs['saturating_der'] = amdata.obs.abs_der<0.0005

amdata.obs['saturating'] = amdata.obs.saturating_std | amdata.obs.saturating_der


amdata.obs.saturating.value_counts()

amdata.write_h5ad(paths.DATA_FITTED)

# %% ########################
# PERSON MODELLING

amdata = amdata_src.AnnMethylData('../../methylation/wave4/wave4_fitted.h5ad')

amdata = amdata[~amdata.obs.saturating]
amdata = amdata[amdata.obs.sort_values('r2').tail(250).index].copy()

modelling_bio.person_model(amdata[:,:10].copy().T, method='advi')

amdata[:,:10].copy().T

with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=partial(modelling_bio.person_model, 
                            method='advi', 
                            show_progress=False),
                iterable=[amdata[:,i] for i in range(1000)],
                chunksize=1
                ), 
            total=amdata[:,:1000].shape[1]))

results
bio_fit = az.summary(results[0],  round_to=5)

processed_results = []
for trace_bio in tqdm(results):    
    bio_fit = az.summary(trace_bio, round_to=5)
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
for param in param_list:
    amdata.var[f'{param}'] = fits.loc[(slice(None),'bio', param)]['mean'].values


ab_maps = modelling_bio.person_model(amdata[:, 0], method='advi')
az.plot_trace(ab_maps)
amdata_chunk = amdata[:,:100].copy()
amdata_chunk.var['acc_norm'] = ab_maps['acc']
amdata_chunk.var['bias_norm'] = ab_maps['bias']

sns.scatterplot(data=amdata_chunk.var, x='acc_cens', y='acc_norm')


sample_pheno = pd.read_csv('../../methylation/wave4/2023-08-02_w4_phenotypes.csv', index_col=0)
amdata.var = amdata.var.join(sample_pheno, on='id')

amdata.var['norm_pack_years'] = np.log(1+amdata.var.pack_years)
# Combine ever_smoke with pack_years
amdata.var['weighted_smoke'] = amdata.var['norm_pack_years']/np.exp(amdata.var['ever_smoke'])

amdata.var.to_csv('wave4_participants.csv')

sns.scatterplot(data=amdata.var, x='acc', y='bias', hue='age')

amdata.write_h5ad(paths.DATA_FITTED)