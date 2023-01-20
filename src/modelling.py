import pymc as pm
import arviz as az

import numpy as np
from scipy.stats import norm

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr


def linear_site(ages, m_values):

    with pm.Model():

        # Define priors
        mean_slope = pm.Uniform("mean_slope",    lower=-1/100,   upper=1/100)
        mean_inter = pm.Uniform("mean_inter",    lower=0,        upper=1)
        var_inter = pm.Uniform("var_inter",     lower=0,        upper=1/10)
        
        # model mean and variance
        mu = mean_slope*ages + mean_inter
        var = var_inter

        # Define likelihood
        likelihood = pm.Normal("m-values",
            mu = mu,
            sigma = np.sqrt(var),
            observed = m_values)

        trace = pm.sample(cores=1, progressbar=False)
        max_p = pm.find_MAP(progressbar=False)

    return trace, max_p

def vect_linear_site(amdata):
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

        pm_data = pm.Data("data", amdata.X.T, dims=("participants", "sites"), mutable=True)

        # Define likelihood
        likelihood = pm.Normal("m-values",
            mu = mean,
            sigma = np.sqrt(variance),
            observed = pm_data)

        trace = pm.sample(cores=1, progressbar=False)
        max_p = pm.find_MAP(progressbar=False)

    return trace, max_p






def drift_site(ages, m_values):
    with pm.Model():

        # Define priors
        mean_slope = pm.Uniform("mean_slope",    lower=-1/100,   upper=1/100)
        mean_inter = pm.Uniform("mean_inter",    lower=0,        upper=1)
        var_slope = pm.Uniform("var_slope",     lower=0,        upper=1/10)
        var_inter = pm.Uniform("var_inter",     lower=0,        upper=1/10)
        
        # model mean and variance
        mu = mean_slope*ages + mean_inter
        var = var_slope*ages + var_inter

        # Define likelihood
        likelihood = pm.Normal("m-values",
            mu = mu,
            sigma = np.sqrt(var) ,
            observed = m_values)

        # trace = pm.sample(cores=1, progressbar=True)

        trace = pm.sample(cores=1, progressbar=False)
        max_p = pm.find_MAP(progressbar=False)

    return trace, max_p

def fit_and_compare(site_data):

    ROUND = 7

    ages = site_data.var.age
    m_values = site_data.X.flatten()
    site_index = site_data.obs.index[0]

    trace_d, map_d = drift_site(ages, m_values)
    trace_l, map_l = linear_site(ages, m_values)
    comparison = az.compare({"drift": trace_d, "linear": trace_l})

    linear_fit = az.summary(trace_l, round_to=ROUND)
    drift_fit = az.summary(trace_d, round_to=ROUND)

    drift_fit.insert(1, 'MAP', np.array(list(map_d.values())[-4:]).round(ROUND))
    linear_fit.insert(1, 'MAP', np.array(list(map_l.values())[-3:]).round(ROUND))

    fit = pd.concat([linear_fit, drift_fit], keys=['linear','drift'], names=['model','param'])
    fit = fit.assign(site=site_index).set_index('site', append=True).reorder_levels(['site','model','param'])

    comparison = comparison.reset_index().rename(columns={'index': 'model'})
    comparison['site'] = site_index
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





def vector_person_model(amdata, show_progress=True):

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

        pm_data = pm.Data("data", amdata.X, dims=("site", "part"), mutable=True)
        
        obs = pm.Normal("obs", mu=mean, sigma = sigma, observed=pm_data)

        trace_ab = pm.sample(1000, tune=1000, chains=4, cores=1, progressbar=show_progress) 
        # map_ab = pm.find_MAP(progressbar=False)

    return trace_ab    

def vectorize_all_participants(amdata):

    # The data has two dimensions: participant and CpG site
    coords = {"site": amdata.sites.index, "part": amdata.participants.index}

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    m_slope = np.broadcast_to(amdata.sites.mean_slope, shape=(amdata.shape[1], amdata.shape[0])).T
    m_int = np.broadcast_to(amdata.sites.mean_inter, shape=(amdata.shape[1], amdata.shape[0])).T
    v_slope = np.broadcast_to(amdata.sites.var_slope, shape=(amdata.shape[1], amdata.shape[0])).T
    v_int = np.broadcast_to(amdata.sites.var_inter, shape=(amdata.shape[1], amdata.shape[0])).T
    age = amdata.participants.age.values

    # Define Pymc model
    with pm.Model(coords=coords) as model:
        
        # Define model variables
        acc = pm.Uniform('acc', lower=-2, upper = 2, dims='part')
        bias = pm.Uniform('bias', lower=-1, upper = 1, dims='part')

        mean = np.exp(acc)*m_slope*age + m_int + bias
        var = v_slope*age + v_int
        sigma = np.sqrt(var)

        pm_data = pm.Data("data", amdata.X, dims=("site", "part"), mutable=True)
        
        obs = pm.Normal("obs", mu=mean, sigma = sigma, observed=pm_data)

        # trace_ab = pm.sample(1000, tune=1000) 
        map_ab = pm.find_MAP(progressbar=False)

    return map_ab

def make_clean_trace(trace):
    delattr(trace, 'log_likelihood')
    delattr(trace, 'sample_stats')
    delattr(trace, 'observed_data')
    delattr(trace, 'constant_data')

def concat_traces(trace1, trace2, dim):
    for group in ['posterior']:
        concatenated_group = xr.concat((trace1[group], trace2[group]), dim=dim)
        setattr(trace1, group, concatenated_group)