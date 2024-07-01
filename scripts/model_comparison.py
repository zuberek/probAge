# %% ########################
### LOADING

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
import pickle
import os

from src import modelling_bio_beta as modelling
from scipy.stats import norm

from pymc.variational.callbacks import CheckParametersConvergence
import pymc as pm
import arviz as az

N_CORES = 15
N_SITES = 1024 # number of sites to take in for the final person inferring

MULTIPROCESSING = True
DATASET_NAME = 'wave3'

# load data
amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_sites_fitted.h5ad', backed='r')
participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_participants.csv', index_col='Basename')

amdata = amdata[amdata.obs.spr2>0.2].to_memory()

# sort values
amdata = amdata[amdata.obs.sort_values('spr2', ascending=False).index]

amdata.obs

# %%

def get_lin_trace(site_data):
    with pm.Model(): 
        sigma = pm.Normal("sigma", 0.01, sigma=0.2)
        intercept = pm.Normal("Intercept", 0.5, sigma=1)
        slope = pm.Normal("slope", 0.1, sigma=2)

        mu=intercept + slope * site_data.var.age

        likelihood = pm.Normal("y", mu=mu, sigma=sigma, 
                            observed=site_data.X.flatten())

        trace_lin = pm.sample(3000, idata_kwargs={"log_likelihood": True}, progressbar=False)
        return trace_lin

def get_bio_trace(site_data):
    site_model = modelling.bio_sites(site_data)
    with site_model:
        trace_bio = pm.sample(1000, idata_kwargs={"log_likelihood": True}, progressbar=False)
    return trace_bio


def get_bio_trace_advi(site_data, n):
    site_model = modelling.bio_sites(site_data)
    with site_model:
        mean_field = pm.fit(method='advi', n=n, callbacks=[CheckParametersConvergence()],  progressbar=False)
        trace_bio = mean_field.sample(1_000)
        pm.compute_log_likelihood(trace_bio)
    return trace_bio


# %% ########################
### COMPUTING

loos = []
for site_index in tqdm(amdata.obs.index[:10]):

    site_data = amdata[site_index]

    if os.path.isfile(f'../exports/traces/{site_index}.idata'):
        with open(f'../exports/traces/{site_index}_lin.idata', 'rb') as f:
            trace_lin = pickle.load(f)
            print(f"Skipping {site_index}_lin.idata")
    else:
        print(f"Computting {site_index}_lin.idata")

        trace_lin = get_lin_trace(site_data)

        # filehandler = open(f'{paths.DATA_PROCESSED_DIR}/traces/{site_index}_lin.idata',"wb")
        # pickle.dump(trace_lin,filehandler)
        # filehandler.close()

    # if os.path.isfile(f'../exports/traces/{site_index}.idata'):
    #     with open(f'../exports/traces/{site_index}.idata', 'rb') as f:
    #         trace_bio = pickle.load(f)
    #         print(f"Skipping {site_index}.idata")
    # else:
    print(f"Computing {site_index}.idata")
    
    trace_bio = get_bio_trace_advi(site_data, n=50_000)

    filehandler = open(f'{paths.DATA_PROCESSED_DIR}/traces/{site_index}.idata',"wb")
    pickle.dump(trace_bio,filehandler)
    filehandler.close()

    loo = az.compare({"bio": trace_bio, "lin": trace_lin})

    loos.append([site_index, loo])

    filehandler = open(f'{paths.DATA_PROCESSED_DIR}/comparisonADVI.pickle',"wb")
    pickle.dump(loos,filehandler)
    filehandler.close()

# %%

# def lin_ll(amdata):
#     # Define priors
#     sigma = pm.Normal("sigma", 0.01, sigma=0.1)
#     intercept = pm.Normal("Intercept", 0.5, sigma=0.5)
#     slope = pm.Normal("slope", 0.1, sigma=1)

#     mu = intercept + slope * age

#     # Define likelihood
#     likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=values)

#     # Inference!
#     # draw 3000 posterior samples using NUTS sampling
#     return norm.logpdf(data, loc=mu, b=sigma).sum(axis=0)


az.plot_trace(idata)
az.summary(idata)

# %% ########################
### MODELLING


site_model = modelling.bio_sites(amdata['cg00529958'])
with site_model:
    mean_field = pm.fit(method='advi', n=50_000, callbacks=[CheckParametersConvergence()],  progressbar=True)
trace = mean_field.sample(1_000)



plt.plot(mean_field.hist)
az.plot_trace(trace)

# with person_model:
#     trace = pm.sample(progressbar=True)

# with person_model:
#     maps = pm.find_MAP(progressbar=True)
# %%
