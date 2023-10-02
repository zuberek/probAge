# %% ########################
# IMPORTING
# %load_ext autoreload 
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

import nutpie
import pymc as pm
import pickle
pm.model.make_initial_point_fn = pm.initial_point.make_initial_point_fn
from src import modelling_bio


n_cores = 50
n_sites = 3_000

# %% ########################
# LOADING
params = modelling_bio.get_site_params()

amdata = amdata_src.AnnMethylData('../exports/wave4_meta.h5ad', backed='r')
amdata = amdata[amdata.obs.sort_values('spr2').tail(n_sites).index]
amdata = amdata.to_memory()

chunk_size = 1
n_sites = amdata.shape[0]
amdata_chunks = []
for i in range(0, n_sites, chunk_size):
    amdata_chunks.append(amdata[i:i+chunk_size])

# %% ########################
# MODEL
def bio_sites(amdata):

    ages = np.broadcast_to(amdata.var.age, shape=(amdata.shape[0], amdata.shape[1])).T
    coords = {'sites': amdata.obs.index.values,
            'participants': amdata.var.index.values}

    with pm.Model(coords=coords) as model:

        data = pm.MutableData("data", amdata.X.T)
        ages = pm.ConstantData("ages", ages)

        # condition on maximal initial standard deviation
        init_std_bound = 0.1

        # Define priors
        eta_0 = pm.Uniform("eta_0", lower=0, upper=1, dims='sites')
        omega = pm.HalfNormal("omega", sigma=0.02, dims='sites')
        p = pm.Uniform("meth_init", lower=0, upper=1, dims='sites')
        N = pm.TruncatedNormal('system_size', mu=100, sigma=100, lower=10, dims='sites')
        var_init = pm.Uniform("var_init", lower=0,
                                          upper=2*np.power(init_std_bound*N,2),
                                          dims='sites')
        
        # Useful variables
        eta_1 = 1-eta_0
        
        # model mean and variance
        var_term_0 = eta_0*eta_1
        var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

        mean = eta_0 + np.exp(-omega*ages)*((p-1)*eta_0 + p*eta_1)

        variance = (var_term_0/N 
                + np.exp(-omega*ages)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*ages)*(var_init/np.power(N,2) - var_term_1/N)
            )

        # Beta function parameter boundaries
        sigma = np.sqrt(variance)
        sigma = pm.math.minimum(sigma,  pm.math.sqrt(mean * (1 - mean)))

        # Define likelihood
        likelihood = pm.Beta("m-values",
            mu = mean,
            sigma = sigma,
            dims=("participants", "sites"),
            observed = data)

    return model

# %% ########################
# MULTIPROCESS
model = bio_sites(amdata_chunks[0])
compiled_model = nutpie.compile_pymc_model(model)

def nutpie_sample(data, cores=1, chains=3):
    # print(f'Starting chunk {data.obs.index[0]}')
    cmodel= compiled_model.with_data(data=data.X.T)
    trace = nutpie.sample(cmodel, target_accept=0.9, cores=cores, chains=chains,
                         progress_bar=False)
    with bio_sites(data):
        pm.compute_log_likelihood(trace)
        
    return trace

with Pool(n_cores, maxtasksperchild=1) as p:
    res = list(tqdm(p.imap(nutpie_sample, amdata_chunks), total=len(amdata_chunks)))

with open(f'../exports/wave4_meta_{n_sites}_fit.pk', 'wb') as f:
    pickle.dump(res, f)

# %% PROCESS RESULTS IN AMDATA

processed_results = []

for trace_bio in tqdm(res):    
    bio_fit = az.summary(trace_bio, var_names=params, round_to=8)
    bio_fit.index = pd.MultiIndex.from_tuples([(index_tuple[1][:-1], 'bio', index_tuple[0]) for index_tuple in bio_fit.index.str.split('[')],
                            names=['site', 'model', 'param'])

    processed_results.append(bio_fit)

fits = pd.DataFrame()
for fit in processed_results:
        fits = pd.concat([fits, fit])

for param in params:
    amdata.obs[f'{param}'] = fits.loc[(slice(None),'bio', param)]['mean'].values

modelling_bio.bio_model_plot(amdata[1])

# save fits information
fits.to_csv('wave4_fits.csv')

amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(amdata, t=90)
amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(amdata)
amdata.obs['saturating_der'] = amdata.obs.abs_der<0.001
amdata.obs['saturating'] = amdata.obs.saturating_std | amdata.obs.saturating_der

# save amdata
amdata.write_h5ad('../exports/wave4_fitted.h5ad')

# %%
