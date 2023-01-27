import pymc as pm
import arviz as az

import numpy as np
from scipy.stats import norm
from pymc.sampling_jax import sample_numpyro_nuts
from pymc.sampling import jax
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr


CHAINS = 1
def linear_sites(amdata, return_MAP=False, return_trace=True, show_progress=False):

    ages = np.broadcast_to(amdata.participants.age, shape=(amdata.n_sites, amdata.n_participants)).T
    coords = {'sites': amdata.sites.index.values,
            'participants': amdata.participants.index.values}

    with pm.Model(coords=coords) as model:

        # Define priors
        mean_slope = pm.Uniform("mean_slope", lower=-1/100, upper=1/100, dims='sites')
        mean_inter = pm.Uniform("mean_inter", lower=0, upper=1, dims='sites')
        var_inter = pm.Uniform("var_inter", lower=0, upper=1/10, dims='sites')
        
        # model mean and variance
        mean = mean_slope*ages + mean_inter
        variance = var_inter

        # Define likelihood
        likelihood = pm.Normal("m_values",
            mu = mean,
            sigma = np.sqrt(variance),
            dims=("participants", "sites"),
            observed = amdata.X.T)

        res = {}
        if return_MAP:
            res['map'] = pm.find_MAP(progressbar=False)

        if return_trace:
            res['trace'] = pm.sample(1000, tune=1000, chains=CHAINS, cores=1, progressbar=show_progress) 
            pm.compute_log_likelihood(res['trace'])
    return res

def bio_sites(amdata, return_MAP=False, return_trace=True, show_progress=False):

    ages = np.broadcast_to(amdata.participants.age, shape=(amdata.n_sites, amdata.n_participants)).T
    coords = {'sites': amdata.sites.index.values,
            'participants': amdata.participants.index.values}

    with pm.Model(coords=coords) as model:

        # Define priors
        nu_0 = pm.Uniform("nu_0", lower=0, upper=0.1, dims='sites')
        nu_1 = pm.Uniform("nu_1", lower=0, upper=0.1 , dims='sites')
        p = pm.Uniform("meth_init", lower=0, upper=1, dims='sites')
        var_init = pm.Uniform("var_init", lower=0, upper=10_000_000, dims='sites')
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

        res = {}
        if return_MAP:
            res['map'] = pm.find_MAP(progressbar=False)

        if return_trace:
            res['trace'] = pm.sample(1000, tune=1000, chains=CHAINS, cores=1, progressbar=show_progress) 
            pm.compute_log_likelihood(res['trace'])

    return res


def fit_and_compare(amdata):

    ROUND = 7

    trace_l = linear_sites(amdata, show_progress=True)['trace']
    trace_bio = bio_sites(amdata, show_progress=True)['trace']

    linear_fit = az.summary(trace_l, round_to=ROUND)
    bio_fit = az.summary(trace_bio, round_to=ROUND)

    linear_fit.index = pd.MultiIndex.from_tuples([(index_tuple[1][:-1], 'linear', index_tuple[0]) for index_tuple in linear_fit.index.str.split('[')],
                              names=['site', 'model', 'param'])
    
    bio_fit.index = pd.MultiIndex.from_tuples([(index_tuple[1][:-1], 'bio', index_tuple[0]) for index_tuple in bio_fit.index.str.split('[')],
                              names=['site', 'model', 'param'])
    all_fits = pd.concat([linear_fit, bio_fit],
                    )
    all_fits.sort_index(level=0)

    all_comparisons = pd.DataFrame()
    for site in amdata.sites.index:
        comparison = az.compare({"linear": trace_l.sel(site=site), "bio": trace_bio.sel(site=site)})
        comparison = comparison.reset_index().rename(columns={'index': 'model'})
        comparison['site'] = site
        comparison = comparison.set_index(['site', 'model'])
        all_comparisons = pd.concat([all_comparisons, comparison])

    return all_fits, all_comparisons



def comparison_postprocess(results, amdata):
    fits = pd.DataFrame()
    comparisons = pd.DataFrame()
    for site in results:
        fit, comparison = site
        fits = pd.concat([fits, fit])
        comparisons = pd.concat([comparisons, comparison])

    amdata.obs['nu_0'] = fits.loc[(slice(None),'bio','nu_0')]['mean'].values
    amdata.obs['nu_1'] = fits.loc[(slice(None),'bio','nu_1')]['mean'].values
    amdata.obs['meth_init'] = fits.loc[(slice(None),'bio','meth_init')]['mean'].values
    amdata.obs['var_init'] = fits.loc[(slice(None),'bio','var_init')]['mean'].values
    amdata.obs['system_size'] = fits.loc[(slice(None),'bio','system_size')]['mean'].values


def plot_confidence(site_index, amdata):
    meta = amdata.obs.loc[site_index]
    x_space = amdata.var.age
    y_mean = x_space*meta.mean_slope + meta.mean_inter
    y_var = x_space*meta.var_slope + meta.var_inter
    y_2pstd = y_mean + 2*(np.sqrt(y_var))
    y_2nstd = y_mean - 2*(np.sqrt(y_var))

    sns.scatterplot(x= amdata.var.age, y=amdata[site_index].X.flatten())
    sns.lineplot(x=x_space, y=y_mean)
    sns.lineplot(x=x_space, y=y_2pstd)
    sns.lineplot(x=x_space, y=y_2nstd)


def accelerated_biased_person_model_map(age, m_values, params):

    with pm.Model():

        # Define priors
        bias = pm.Uniform("bias", lower=-1, upper=1)
        acc = pm.Uniform("acc", lower=-2, upper=2)
        
        mu = np.exp(acc)*params.mean_slope*age + params.mean_inter + bias
        var = params.var_slope*age + params.var_inter

        # Define likelihood
        likelihood = pm.Normal("m-values",
            mu=mu,
            sigma=np.sqrt(var),
            observed=m_values)

        map=pm.find_MAP(progressbar=True)

    return map


def fit_person(person_data):
    age = person_data.var.age[0]
    m_values = person_data.X.flatten()
    params = person_data.obs[['mean_slope', 'mean_inter', 'var_slope', 'var_inter']]
    person_index = person_data.var.index[0]

    trace = accelerated_biased_person_model(age, m_values, params)

    fit = az.summary(trace, round_to=5)
    # fit.insert(1, 'MAP', np.array(list(map.values())[-2:]).round(5))
    fit = fit.reset_index().rename(columns={'index': 'param'})
    fit = fit.assign(person=person_index).set_index(['person','param'])

    return fit

def fit_person_MAP(person_data):
    # person_data = amdata_T[0]
    age = person_data.obs.age[0]
    m_values = person_data.X.flatten()
    params = person_data.var[['mean_slope', 'mean_inter', 'var_slope', 'var_inter']]
    map = accelerated_biased_person_model_map(age, m_values, params)
    return map


def person_model(amdata, return_trace=True, return_MAP=False, show_progress=False):

    # The data has two dimensions: participant and CpG site
    coords = {"site": amdata.sites.index, "part": amdata.participants.index}

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    nu_0 = np.broadcast_to(amdata.sites.nu_0, shape=(amdata.shape[1], amdata.shape[0])).T
    nu_1 = np.broadcast_to(amdata.sites.nu_1, shape=(amdata.shape[1], amdata.shape[0])).T
    p = np.broadcast_to(amdata.sites.meth_init, shape=(amdata.shape[1], amdata.shape[0])).T
    var_init = np.broadcast_to(amdata.sites.var_init, shape=(amdata.shape[1], amdata.shape[0])).T
    N = np.broadcast_to(amdata.sites.system_size, shape=(amdata.shape[1], amdata.shape[0])).T

    age = amdata.participants.age.values


    # Define Pymc model
    with pm.Model(coords=coords) as model:
        
        # Define model variables
        acc = pm.Uniform('acc', lower=-2, upper = 2, dims='part')
        bias = pm.Uniform('bias', lower=-1, upper = 1, dims='part')

        # Useful variables
        exp_acc = np.exp(acc)
        acc_nu_0 = exp_acc*nu_0
        acc_nu_1 = exp_acc*nu_1
        omega = acc_nu_0 + acc_nu_1
        eta_0 = acc_nu_0/omega
        eta_1 = acc_nu_1/omega
        
        # model mean and variance
        var_term_0 = eta_0*eta_1
        var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

        mean = eta_0 + np.exp(-omega*age)*((p-1)*eta_0 + p*eta_1) + bias
        mean = pm.math.minimum(mean, 1)
        mean = pm.math.maximum(mean,0)

        variance = (var_term_0/N 
                + np.exp(-omega*age)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*age)*(var_init/np.power(N,2) - var_term_1/N)
            )

        # Define likelihood
        obs = pm.Normal("obs", mu=mean,
                               sigma = np.sqrt(variance), 
                               dims=("site", "part"), 
                               observed=amdata.X)

        res = {}
        if return_MAP:
            res['map'] = pm.find_MAP(progressbar=False)

        if return_trace:
            res['trace'] = pm.sample(1000, tune=1000, chains=4, cores=1, progressbar=show_progress) 

    return res    

def make_clean_trace(trace):
    delattr(trace, 'log_likelihood')
    delattr(trace, 'sample_stats')
    delattr(trace, 'observed_data')
    delattr(trace, 'constant_data')

def concat_traces(trace1, trace2, dim):
    for group in ['posterior']:
        concatenated_group = xr.concat((trace1[group], trace2[group]), dim=dim)
        setattr(trace1, group, concatenated_group)