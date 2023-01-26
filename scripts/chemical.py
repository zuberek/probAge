import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import modelling
import pymc as pm
import arviz as az
from pymc.sampling_jax import sample_numpyro_nuts

# import logging
# logger = logging.getLogger('pymc')
# logger.propagate = False
# logger.setLevel(logging.ERROR)

N_SITES =  10

wave3 = amdata.AnnMethylData('../exports/wave3_linear.h5ad')
wave3 = wave3[wave3.sites.sort_values('r2', ascending=False).index[:N_SITES]]
# %%
def vect_linear_site(amdata):
    ages = np.broadcast_to(amdata.participants.age, shape=(amdata.n_sites, amdata.n_participants)).T
    coords = {'sites': amdata.sites.index.values,
            'participants': amdata.participants.index.values}
    print('lemon')

    with pm.Model(coords=coords) as model:

        # Define priors
        mean_slope = pm.Uniform("mean_slope", lower=-1/100, upper=1/100, dims='sites')
        mean_inter = pm.Uniform("mean_inter", lower=0, upper=1, dims='sites')
        var_inter = pm.Uniform("var_inter", lower=0, upper=1/10, dims='sites')
        
        # model mean and variance
        mean = mean_slope*ages + mean_inter
        variance = var_inter

        pm_data = pm.Data("data", amdata.X.T, dims=("participants", "sites"), mutable=True)

        # Define likelihood
        likelihood = pm.Normal("m-values",
            mu = mean,
            sigma = np.sqrt(variance),
            dims=("participants", "sites"),
            observed = amdata.X.T)

        trace = sample_numpyro_nuts(postprocessing_backend='cpu',
                                    progressbar=True)
        pm.compute_log_likelihood(trace)

    return trace

def vect_chem_site(amdata):
    ages = np.broadcast_to(amdata.participants.age, shape=(amdata.n_sites, amdata.n_participants)).T
    coords = {'sites': amdata.sites.index.values,
            'participants': amdata.participants.index.values}
            
    with pm.Model(coords=coords) as model:

        # Define priors
        nu_0 = pm.Uniform("nu_0", lower=0, upper=0.1, dims='sites')
        nu_1 = pm.Uniform("nu_1", lower=0, upper=0.1, dims='sites')
        p = pm.Uniform("init_meth", lower=0, upper=1, dims='sites')
        var_init = pm.Uniform("var_init", lower=0, upper=1_000_000, dims='sites')
        N = pm.Uniform('system_size', lower= 1, upper=1_000_1000, dims='sites')


        # Useful variables
        omega = nu_0 + nu_1
        eta_0 = nu_0/omega
        eta_1 = nu_1/omega
        
        # model mean and variance
        var_term_0 = eta_0*eta_1
        var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

        mean = eta_0 + np.exp(-omega*ages)*((p-1)*eta_0 + p*eta_1)

        variance = (var_term_0/N 
                + np.exp(-omega*ages)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*ages)*(var_init/np.power(N,2) - var_term_1/N)
            )

        # Define likelihood
        likelihood = pm.Beta("m-values",
            mu = mean,
            sigma = np.sqrt(variance),
            dims=("participants", "sites"),
            observed = amdata.X.T)

        trace = sample_numpyro_nuts(postprocessing_backend = 'cpu',
                                    progressbar=True)

        pm.compute_log_likelihood(trace)
        # max_p = pm.find_MAP(progressbar=True)

    return trace
for i in range(4):
    trace_chem = vect_chem_site(wave3[:2])
trace_linear = vect_linear_site(wave3[:2])

import pickle 

with open('trace_chem.pk', 'wb') as f:
    pickle.dump(trace_chem, f)

with open('trace_linear.pk', 'wb') as f:
    pickle.dump(trace_linear, f)