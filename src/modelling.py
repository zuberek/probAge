import pymc as pm
import arviz as az

import numpy as np
from scipy.stats import norm

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr


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
            res['trace'] = pm.sample(1000, tune=1000, chains=4, cores=1, progressbar=show_progress) 

    return res    


def drift_sites(amdata, return_MAP=False, return_trace=True, show_progress=False):
    ages = np.broadcast_to(amdata.participants.age, shape=(amdata.n_sites, amdata.n_participants)).T
    coords = {'sites': amdata.sites.index.values,
            'participants': amdata.participants.index.values}

    with pm.Model(coords=coords) as model:

        # Define priors
        mean_slope = pm.Uniform("mean_slope", lower=-1/100, upper=1/100, dims='sites')
        mean_inter = pm.Uniform("mean_inter", lower=0, upper=1, dims='sites')
        var_slope = pm.Uniform("var_slope",     lower=0,        upper=1/10)
        var_inter = pm.Uniform("var_inter", lower=0, upper=1/10, dims='sites')
        
        # model mean and variance
        mean = mean_slope*ages + mean_inter
        variance = var_slope*ages + var_inter

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
            res['trace'] = pm.sample(1000, tune=1000, chains=4, cores=1, progressbar=show_progress) 

    return res    

def fit_and_compare(amdata):

    ROUND = 5

    drift_trace = drift_sites(amdata)['trace']
    linear_trace = linear_sites(amdata)['trace']

    comparison = az.compare({"drift": drift_trace, "linear": linear_trace})

    linear_fit = az.summary(linear_trace, round_to=ROUND)
    drift_fit = az.summary(drift_trace, round_to=ROUND)

    # drift_fit.insert(1, 'MAP', np.array(list(map_d.values())[-4:]).round(ROUND))
    # linear_fit.insert(1, 'MAP', np.array(list(map_l.values())[-3:]).round(ROUND))

    fit = pd.concat([linear_fit, drift_fit], keys=['linear','drift'], names=['model','param'])
    # fit = fit.assign(site=site_index).set_index('site', append=True).reorder_levels(['site','model','param'])

    comparison = comparison.reset_index().rename(columns={'index': 'model'})
    comparison['site'] = amdata.obs.index
    comparison = comparison.set_index(['site', 'model'])

    # return fit, comparison, trace_d, trace_l

    return fit, comparison

def comparison_postprocess(results, amdata):
    fits = pd.DataFrame()
    comparisons = pd.DataFrame()
    for site in results:
        fit, comparison = site
        fits = pd.concat([fits, fit])
        comparisons = pd.concat([comparisons, comparison])

    amdata.obs['mean_slope'] = fits.loc[(slice(None),'drift','mean_slope')]['mean'].values
    amdata.obs['mean_inter'] = fits.loc[(slice(None),'drift','mean_inter')]['mean'].values
    amdata.obs['var_slope'] = fits.loc[(slice(None),'drift','var_slope')]['mean'].values
    amdata.obs['var_inter'] = fits.loc[(slice(None),'drift','var_inter')]['mean'].values

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



def person_model(amdata, return_trace=True, return_MAP=False, show_progress=False):

    # The data has two dimensions: participant and CpG site
    coords = {"site": amdata.sites.index, "part": amdata.participants.index}

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    m_slope = np.broadcast_to(amdata.sites.mean_slope, shape=(amdata.shape[1], amdata.shape[0])).T
    m_int = np.broadcast_to(amdata.sites.mean_inter, shape=(amdata.shape[1], amdata.shape[0])).T
    v_slope = np.broadcast_to(amdata.sites.var_slope, shape=(amdata.shape[1], amdata.shape[0])).T
    v_int = np.broadcast_to(amdata.sites.var_inter, shape=(amdata.shape[1], amdata.shape[0])).T
    age = amdata.participants.age.values

    if show_progress: print('lemon')

    # Define Pymc model
    with pm.Model(coords=coords) as model:
        
        # Define model variables
        acc = pm.Uniform('acc', lower=-2, upper = 2, dims='part')
        bias = pm.Uniform('bias', lower=-1, upper = 1, dims='part')

        mean = np.exp(acc)*m_slope*age + m_int + bias
        var = v_slope*age + v_int
        sigma = np.sqrt(var)

        obs = pm.Normal("obs", mu=mean, sigma = sigma, dims=("site", "part"), observed=amdata.X)

        res = {}
        if return_MAP:
            res['map'] = pm.find_MAP(progressbar=False)

        if return_trace:
            res['trace'] = pm.sample(1000, tune=1000, chains=4, cores=1, progressbar=show_progress) 

    return res    

def make_clean_trace(trace):
    # delattr(trace, 'log_likelihood')
    # delattr(trace, 'sample_stats')
    delattr(trace, 'observed_data')
    delattr(trace, 'constant_data')

def concat_traces(trace1, trace2, dim):
    for group in ['posterior']:
        concatenated_group = xr.concat((trace1[group], trace2[group]), dim=dim)
        setattr(trace1, group, concatenated_group)