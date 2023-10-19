# %% ########################
# IMPORTING
# %load_ext autoreload 
# %autoreload 2

import sys
jobname = sys.argv[1]
num1 = int(sys.argv[2])
num2 = int(sys.argv[3])
print(num1)
print(num2)

#jobname = "EWAS"
#num1 = 500
#num2 = 1000

sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

import nutpie
import pymc as pm
import pickle
pm.model.make_initial_point_fn = pm.initial_point.make_initial_point_fn
from src import modelling_bio

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

n_cores = 10
n_sites = 100

# %% ########################
# LOADING
params = modelling_bio.get_site_params()

if jobname == "EWAS":
    amdata = amdata_src.AnnMethylData('../exports/EWAS/ewasKNN_fitted.h5ad', backed='r')
elif jobname == "wave4":
    amdata = amdata_src.AnnMethylData('../exports/wave4/wave4_meta.h5ad', backed='r')
else:
    # Handle the case when job name is neither "ewas" nor "wave4"
    raise ValueError("Invalid job name")

amdata = amdata[amdata.obs.sort_values('spr2', ascending=False).index[num1:num2]]
amdata = amdata.to_memory()

chunk_size = 1
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
# define model in pymc
model = bio_sites(amdata_chunks[0])
# compile model in nutpie
compiled_model = nutpie.compile_pymc_model(model)

def nutpie_sample(data, jobname, cores=1, chains=3):
    if jobname == "EWAS":
        return modelling_bio.bio_sites(data, method='nuts', nuts_sampler='pymc', show_progress=False)
    elif jobname == "wave4":
        # switching data to compiled nutpie model
        cmodel = compiled_model.with_data(data=data.X.T)
        cmodel.coords['site'] = data.obs.index
        # compute trace
        trace = nutpie.sample(cmodel, target_accept=0.8, cores=cores, chains=chains, progress_bar=True)
        # compute log_likelihood
        with bio_sites(data):
            pm.compute_log_likelihood(trace)
        return trace
    else:
        raise ValueError("Invalid job name")

partial_nutpie_sample = partial(nutpie_sample, jobname=jobname)
with Pool(n_cores, maxtasksperchild=1) as p:
    res = list(tqdm(p.imap(partial_nutpie_sample, amdata_chunks), total=len(amdata_chunks)))

print("res made")

#modelling_bio.bio_model_plot(amdata[0])

with open(f'../exports/{jobname}/{jobname}_traces_{num1}to{num2}.pk', 'wb') as f:
    pickle.dump(res, f)
print("res saved")
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
fits.to_csv(f'../exports/{jobname}/{jobname}_{num1}to{num2}_fits.csv')
print('fit saved')

amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(amdata, t=90)
amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(amdata)
amdata.obs['saturating_der'] = amdata.obs.abs_der<0.001
amdata.obs['saturating'] = amdata.obs.saturating_std | amdata.obs.saturating_der

# save amdata
amdata.write_h5ad(f'../exports/{jobname}/{jobname}_{num1}to{num2}_fitted.h5ad')

# %%