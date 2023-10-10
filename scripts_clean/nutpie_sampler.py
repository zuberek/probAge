# %% ########################
# IMPORTING
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

import nutpie
import pymc as pm
from src import modelling_bio



# %% ########################
# LOADING
params = modelling_bio.get_site_params()

amdata = amdata_src.AnnMethylData('../exports/wave4_meta.h5ad', backed='r')
amdata = amdata[amdata.obs.sort_values('spr2').tail(100).index]
amdata = amdata.to_memory()

chunk_size = 2
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
        N = pm.Uniform('system_size', lower= 1, upper=2_000, dims='sites')
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
# MODEL

def ingest_trace(amdata, trace):
    bio_fit = az.summary(trace, var_names=params, round_to=7)
    bio_fit.index = pd.MultiIndex.from_tuples([(index_tuple[1][:-1], 'bio', index_tuple[0]) for index_tuple in bio_fit.index.str.split('[')],
                            names=['site', 'model', 'param'])
    param_list = bio_fit.xs('bio', level='model').index.get_level_values(level='param')
    for param in param_list:
        amdata.obs[f'{param}'] = bio_fit.loc[(slice(None),'bio', param)]['mean'].values



# %% ########################
# SEQUENTIAL

# compile for site 1
site1 = amdata[:10]
model = bio_sites(site1)
compiled_model = nutpie.compile_pymc_model(model)

# sample site 1
trace_pymc = nutpie.sample(compiled_model, save_warmup=False, target_accept=0.9)
az.plot_trace(trace_pymc, var_names=params)
az.plot_posterior(trace_pymc, var_names=params)
az.summary(trace_pymc, var_names=params, round_to=7)

ingest_trace(site1, trace_pymc)
modelling_bio.bio_model_plot(site1)


# switch sites
site2 = amdata[5]
sns.scatterplot(x=amdata.var.age, y=amdata[5].X.flatten())
compiled_model = compiled_model.with_data(data=site2.X.T)

# sample site 2
trace_pymc2 = nutpie.sample(compiled_model, save_warmup=False, target_accept=0.9)
az.plot_trace(trace_pymc2, var_names=params)
az.plot_posterior(trace_pymc2, var_names=params)
az.summary(trace_pymc2, var_names=params, round_to=7)

ingest_trace(site2, trace_pymc2)
modelling_bio.bio_model_plot(site2)


# %% ########################
# MULTIPROCESS
model = bio_sites(amdata_chunks[0])
compiled_model = nutpie.compile_pymc_model(model)

def nutpie_sample(data, cores=1, chains=3):
    print(f'Starting chunk {amdata.obs.index[0]}')
    cmodel= compiled_model.with_data(data=data.X.T)
    return nutpie.sample(cmodel, target_accept=0.9, cores=cores, chains=chains,
                         progress_bar=True)

nutpie_sample(amdata_chunks[0], cores=4, chains=1)

with Pool(15) as p:
    res = list(tqdm(p.imap(nutpie_sample, amdata_chunks[:15]), total=len(amdata_chunks[:15])))

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

# save fits information
fits.to_csv('wave4_fits.csv')
# save amdata
amdata.write_h5ad('../exports/wave4_fitted.h5ad')
