# %% ########################
### LOADING

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
import pickle

from src import modelling_bio_beta as model
from scipy.stats import norm

from pymc.variational.callbacks import CheckParametersConvergence
import pymc as pm
import arviz as az

N_CORES = 15
N_SITES = 1024 # number of sites to take in for the final person inferring

MULTIPROCESSING = True
DATASET_NAME = 'wave3'

# load data
amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_person_fitted.h5ad')
participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_participants.csv', index_col='Basename')

# sort values
amdata = amdata[amdata.obs.sort_values('spr2', ascending=False).index]

# plot top sites
top_sites_indexes = amdata[:12].obs.index
axs = plot.tab(top_sites_indexes, ncols=4)
for i, index in enumerate(top_sites_indexes):
    model.bio_model_plot(amdata[index], ax=axs[i])



site_data = amdata[top_sites_indexes[0]]
age = site_data.var.age.values
values = site_data.X.flatten()

values.shape

model.bio_model_plot(site_data)

site_index = top_sites_indexes[0]

loos = []
for site_index in tqdm(top_sites_indexes[:100]):

    site_data = amdata[site_index]

    with pm.Model() as model_lin: 
        sigma = pm.Normal("sigma", 0.01, sigma=0.1)
        intercept = pm.Normal("Intercept", 0.5, sigma=0.5)
        slope = pm.Normal("slope", 0.1, sigma=1)

        mu=intercept + slope * site_data.var.age

        likelihood = pm.Normal("y", mu=mu, sigma=sigma, 
                            observed=site_data.X.flatten())

        trace_lin = pm.sample(3000, idata_kwargs={"log_likelihood": True}, progressbar=False)

    site_model = model.bio_sites(site_data)
    with site_model:
        mean_field = pm.fit(method='advi', n=50_000, callbacks=[CheckParametersConvergence()],  progressbar=False)
        trace_bio = mean_field.sample(1_000)
        pm.compute_log_likelihood(trace_bio)


    bio_model = model.bio_sites(site_data)
    with bio_model:
        trace_bio = pm.sample(3000, idata_kwargs={"log_likelihood": True})

    loo = az.compare({"bio": trace_bio, "lin": trace_lin})

    loos.append(loo)

    filehandler = open(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}/{site_index}_lin.idata',"wb")
    pickle.dump(trace_lin,filehandler)
    filehandler.close()

    filehandler = open(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}/{site_index}_bio.idata',"wb")
    pickle.dump(trace_bio,filehandler)
    filehandler.close()


filehandler = open(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}/comparison.pickle',"wb")
pickle.dump(loos,filehandler)
filehandler.close()


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


site_model = model.bio_sites(amdata[top_sites_indexes])
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
