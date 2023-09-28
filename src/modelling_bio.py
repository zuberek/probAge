import pymc as pm
import arviz as az

import numpy as np
from scipy.stats import norm, beta
from scipy.optimize import minimize
from pymc.variational.callbacks import CheckParametersConvergence
# from pymc.sampling_jax import sample_numpyro_nuts
# from pymc.sampling import jax
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import xarray as xr

from tqdm import tqdm
from operator import attrgetter

# import color palette
import sys
sys.path.append("..") 
# from src.general_imports import sns_colors

from src.utils import plot

CHAINS = 4
CORES = 1
# cores are set at 1 to allow to 
# avoid daemonic children in mp

SITE_PARAMETERS = {
    'eta_0':    'eta_0',
    'omega':    'omega',
    'p':        'meth_init',
    'N':        'system_size',
    'var_init': 'var_init'
}

def get_site_params():
    return list(SITE_PARAMETERS.values())

def chunkify_sites(amdata, N_CORES=15):
    chunk_size = (amdata.shape[0]//N_CORES)
    n_sites = amdata.shape[0]
    amdata_chunks = []
    for i in range(0, n_sites, chunk_size):
        amdata_chunks.append(amdata[i:i+chunk_size].copy())
    return amdata_chunks


def linear_sites(amdata, return_MAP=False, return_trace=True, cores=CORES, show_progress=False):

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
            res['trace'] = pm.sample(1000, tune=1000,
                            chains=CHAINS, cores=cores,
                            progressbar=show_progress) 
            pm.compute_log_likelihood(res['trace'], progressbar=show_progress)

    return res

def bio_site_mean(ages, eta_0, omega, p):
    return eta_0 + np.exp(-omega*ages)*(p-eta_0)

def bio_sites(amdata, method='MAP', show_progress=True, nuts_sampler='nutpie', init_nuts='auto', target_accept=0.8, cores=CORES, chains=CHAINS):

    if show_progress: print(f'Computing {method} of {amdata.shape[0]} bio_sites')
    ages = np.broadcast_to(amdata.var.age, shape=(amdata.shape[0], amdata.shape[1])).T
    coords = {'sites': amdata.obs.index.values,
            'participants': amdata.var.index.values}

    with pm.Model(coords=coords) as model:

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
        mean = pm.math.where(mean>1, 0.999, mean)
        mean = pm.math.where(mean<0, 0.001, mean)

        # mean = bio_site_mean(ages, eta_0, omega, p)


        variance = (var_term_0/N 
                + np.exp(-omega*ages)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*ages)*(var_init/np.power(N,2) - var_term_1/N)
            )

        # Beta function parameter boundaries
        sigma = np.sqrt(variance)
        # sigma = pm.math.minimum(sigma,  pm.math.sqrt(mean * (1 - mean)) - 0.0001)

        # Define likelihood
        likelihood = pm.Beta("m-values",
            mu = mean,
            sigma = sigma,
            dims=("participants", "sites"),
            observed = amdata.X.T)


        res = {}
        if method=='map':
            return pm.find_MAP(progressbar=show_progress)

        if method=='advi':
            mean_field = pm.fit(method='advi', n=100_000, callbacks=[CheckParametersConvergence()],  progressbar=show_progress)
            return mean_field.sample(1_000)

        if method=='nuts':
            trace = pm.sample(1000, tune=1000, init=init_nuts,
                                     chains=chains, cores=cores,
                                     progressbar=show_progress,
                                     target_accept=target_accept, 
                                     nuts_sampler=nuts_sampler)

            pm.compute_log_likelihood(trace, progressbar=False)
            ppc = pm.sample_posterior_predictive(trace,
                    extend_inferencedata=True)
            return trace


def bio_model(amdata):

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
        # mean = bio_site_mean(ages, eta_0, omega, p)

        variance = (var_term_0/N 
                + np.exp(-omega*ages)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*ages)*(var_init/np.power(N,2) - var_term_1/N)
            )

        # Beta function parameter boundaries
        sigma = np.sqrt(variance)
        sigma = pm.math.minimum(sigma,  pm.math.sqrt(mean * (1 - mean)) - 0.0001)

        # Define likelihood
        likelihood = pm.Beta("m-values",
            mu = mean,
            sigma = sigma,
            dims=("participants", "sites"),
            observed = data)

    return model

def bio_model_fit(model, amdata_chunk, return_MAP=False, return_trace=True, show_progress=False, init_nuts='auto', target_accept=0.8, cores=CORES):

    with model:

        pm.set_data({"data": amdata_chunk.X.T})

        res = {}
        if return_MAP:
            res['map'] = pm.find_MAP(progressbar=show_progress)

        if return_trace:
            trace = pm.sample(1000, tune=1000, init=init_nuts,
                                     chains=CHAINS, cores=cores,
                                     progressbar=show_progress,
                                     target_accept=target_accept, 
                                     nuts_sampler='nutpie')

            pm.compute_log_likelihood(trace, progressbar=False)
            ppc = pm.sample_posterior_predictive(trace,
                    extend_inferencedata=True)
            res['trace'] = trace

    return res

def person_model(amdata,
                         return_trace=True,
                         return_MAP=False,
                         show_progress=False,
                         init_nuts='auto',
                         cores=CORES,
                         nuts_sampler='pymc',
                         map_method='L-BFGS-B'):

    if show_progress: print(f'Modelling {amdata.shape[1]} participants')

    # The data has two dimensions: participant and CpG site
    coords = {"site": amdata.obs.index, "part": amdata.var.index}

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    omega = np.broadcast_to(amdata.obs.omega, shape=(amdata.shape[1], amdata.shape[0])).T
    eta_0 = np.broadcast_to(amdata.obs.eta_0, shape=(amdata.shape[1], amdata.shape[0])).T
    p = np.broadcast_to(amdata.obs.meth_init, shape=(amdata.shape[1], amdata.shape[0])).T
    var_init = np.broadcast_to(amdata.obs.var_init, shape=(amdata.shape[1], amdata.shape[0])).T
    N = np.broadcast_to(amdata.obs.system_size, shape=(amdata.shape[1], amdata.shape[0])).T

    age = amdata.var.age.values


    # Define Pymc model
    with pm.Model(coords=coords) as model:

        
        
        # Define model variables
        acc = pm.Uniform('acc', lower=0.33, upper=3, dims='part')
        bias = pm.Normal('bias', mu=0, sigma=0.1, dims='part')

        # Useful variables
        omega = acc*omega
        eta_1 = 1-eta_0
        
        # model mean and variance
        var_term_0 = eta_0*eta_1
        var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)


        # mean = eta_0 + np.exp(-omega*age)*((p-1)*eta_0 + p*eta_1)
        mean = eta_0 + np.exp(-omega*age)*((p-1)*eta_0 + p*eta_1) + bias

        variance = (var_term_0/N 
                + np.exp(-omega*age)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*age)*(var_init/np.power(N,2) - var_term_1/N)
            )
        
        # Beta function parameter boundaries
        sigma = np.sqrt(variance)
        # sigma = pm.math.minimum(sigma,  pm.math.sqrt(mean * (1 - mean)) - 0.0001)



        # Define likelihood
        # y_latent = pm.Normal.dist(mu=mean, sigma=sigma)
        # obs = pm.Censored("obs",
        #                   y_latent,
        #                   lower=0, upper=1,
        #                   observed=amdata.X)
        
        # # Define likelihood
        obs = pm.Normal("obs",
                            mu=mean,
                            sigma = sigma, 
                            # lower=0, upper=1,
                            dims=("site", "part"), 
                            observed=amdata.X)

        res = {}
        if return_MAP:
            res['map'] = pm.find_MAP(progressbar=show_progress,
                                     method=map_method,
                                     maxeval=10_000)
            res['map']['acc'] = np.log2(res['map']['acc'])

        if return_trace:
            res['trace'] = pm.sample(1000, tune=1000, init=init_nuts,
                                    chains=CHAINS, cores=cores, nuts_sampler=nuts_sampler,
                                    progressbar=show_progress)

    return res    

def person_model(amdata,
                         method='map',
                         show_progress = True,
                         init_nuts='auto',
                         cores=CORES,
                         target_accept=0.9,
                         nuts_sampler='pymc',
                         map_method='L-BFGS-B'):

    
    # if 'age' in amdata.obs.columns:
    #     print('got here')
    #     amdata = amdata.copy().T

    if show_progress: print(f'Modelling {amdata.shape[1]} participants')

    # The data has two dimensions: participant and CpG site
    coords = {"site": amdata.obs.index, "part": amdata.var.index}

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    omega = np.broadcast_to(amdata.obs.omega, shape=(amdata.shape[1], amdata.shape[0])).T
    eta_0 = np.broadcast_to(amdata.obs.eta_0, shape=(amdata.shape[1], amdata.shape[0])).T
    p = np.broadcast_to(amdata.obs.meth_init, shape=(amdata.shape[1], amdata.shape[0])).T
    var_init = np.broadcast_to(amdata.obs.var_init, shape=(amdata.shape[1], amdata.shape[0])).T
    N = np.broadcast_to(amdata.obs.system_size, shape=(amdata.shape[1], amdata.shape[0])).T

    age = amdata.var.age.values

    # Define Pymc model
    with pm.Model(coords=coords) as model:

        # Define model variables
        acc = pm.Uniform('acc', lower=0.33, upper=3, dims='part')
        bias = pm.Normal('bias', mu=0, sigma=0.1, dims='part')

        # Useful variables
        omega = acc*omega
        eta_1 = 1-eta_0
        
        # model mean and variance
        var_term_0 = eta_0*eta_1
        var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)


        # mean = eta_0 + np.exp(-omega*age)*((p-1)*eta_0 + p*eta_1)
        mean = eta_0 + np.exp(-omega*age)*((p-1)*eta_0 + p*eta_1) + bias

        variance = (var_term_0/N 
                + np.exp(-omega*age)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*age)*(var_init/np.power(N,2) - var_term_1/N)
            )
        
        # Beta function parameter boundaries
        sigma = np.sqrt(variance)
        # sigma = pm.math.minimum(sigma,  pm.math.sqrt(mean * (1 - mean)) - 0.0001)

        obs = pm.Beta("obs",
                            mu=mean,
                            sigma = sigma, 
                            # lower=0, upper=1,
                            dims=("site", "part"), 
                            observed=amdata.X)

        if method=='map':
            return pm.find_MAP(progressbar=show_progress)

        if method=='advi':
            mean_field = pm.fit(method='advi', n=100_000, callbacks=[CheckParametersConvergence()],  progressbar=show_progress)
            return mean_field.sample(10_000)

        if method=='nuts':
            trace = pm.sample(1000, tune=1000, init=init_nuts,
                                     chains=CHAINS, cores=cores,
                                     progressbar=show_progress,
                                     target_accept=target_accept, 
                                     nuts_sampler='nutpie')

            pm.compute_log_likelihood(trace, progressbar=False)
            ppc = pm.sample_posterior_predictive(trace,
                    extend_inferencedata=True)
            return trace


    return res    


def fit_and_compare(amdata, cores=CORES, show_progress=False):

    ROUND = 7

    trace_l = linear_sites(amdata,
                            show_progress=False, cores=cores)['trace']
    trace_bio = bio_sites(amdata, init_nuts='advi+adapt_diag',
                            show_progress=show_progress, cores=cores)['trace']

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

    # Extract parameter names from bio model
    param_list = fit.xs('bio', level='model').index.get_level_values(level='param')
    for param in param_list:
        amdata.obs[param] = fits.loc[(slice(None),'bio', param)]['mean'].values
    
    # Detect saturating
    saturating_list = []
    for site in amdata:
        saturating_list.append(
            is_saturating(
                site))

    amdata.obs['saturating'] = saturating_list

    return fits, comparisons

def bio_fit(amdata, show_progress=False, cores=CORES):
    ROUND = 7
    trace_bio = bio_sites(amdata, return_trace=True, return_MAP=False, cores=cores, show_progress=show_progress)['trace']
    bio_fit = az.summary(trace_bio, round_to=ROUND)
    bio_fit.index = pd.MultiIndex.from_tuples([(index_tuple[1][:-1], 'bio', index_tuple[0]) for index_tuple in bio_fit.index.str.split('[')],
                            names=['site', 'model', 'param'])

    loo = az.loo(trace_bio)
    loo = loo['elpd_loo']

    return bio_fit, loo

def bio_fit_post(results, amdata):
    fits = pd.DataFrame()
    loo_list = []
    for site in results:
            fit, loo = site
            fits = pd.concat([fits, fit])
            loo_list.append(loo)
    # Save elpd_loo
    amdata.obs['elpd_loo'] = loo_list
    # Extract parameter names from bio model
    param_list = fit.xs('bio', level='model').index.get_level_values(level='param')
    for param in param_list:
        amdata.obs[param] = fits.loc[(slice(None),'bio', param)]['mean'].values

    return fits



def make_clean_trace(trace):
    delattr(trace, 'sample_stats')
    delattr(trace, 'observed_data')

def concat_traces(trace1, trace2, dim):
    for group in ['posterior']:
        concatenated_group = xr.concat((trace1[group], trace2[group]), dim=dim)
        setattr(trace1, group, concatenated_group)


def bio_model_stats(amdata, t, acc=0, bias=0):
    """Extract mean and variace of site at a given set of 
    time-points."""

    # Extract parameters from site
    eta_0 = amdata.obs['eta_0'].to_numpy()
    omega = amdata.obs['omega'].to_numpy()
    var_init = amdata.obs['var_init'].to_numpy()
    p = amdata.obs['meth_init'].to_numpy()
    N = amdata.obs['system_size'].to_numpy()

    # update acc and bias
    omega = np.exp(acc)*omega
    # reparametrization
    eta_1 = 1-eta_0

    # model mean and variance
    var_term_0 = eta_0*eta_1
    var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

    mean = eta_0 + np.exp(-omega*t)*((p-1)*eta_0 + p*eta_1)

    #update bias
    mean = mean + bias

    variance = (var_term_0/N 
            + np.exp(-omega*t)*(var_term_1-var_term_0)/N 
            + np.exp(-2*omega*t)*(var_init/np.power(N,2) - var_term_1/N)
        )
    return mean, variance

def bio_model_stats_vect(amdata, t, acc=0, bias=0):
    """Extract mean and variace of site at a given set of 
    time-points."""

    # Extract parameters from site
    eta_0 = np.broadcast_to(amdata.obs.eta_0, shape=(len(t), amdata.shape[0])).T
    omega = np.broadcast_to(amdata.obs.omega, shape=(len(t), amdata.shape[0])).T
    var_init = np.broadcast_to(amdata.obs.var_init, shape=(len(t), amdata.shape[0])).T
    p = np.broadcast_to(amdata.obs.meth_init, shape=(len(t), amdata.shape[0])).T
    N = np.broadcast_to(amdata.obs.system_size, shape=(len(t), amdata.shape[0])).T

    # update acc and bias
    omega = np.exp(acc)*omega
    # reparametrization
    eta_1 = 1-eta_0

    # model mean and variance
    var_term_0 = eta_0*eta_1
    var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

    mean = eta_0 + np.exp(-omega*t)*((p-1)*eta_0 + p*eta_1)

    #update bias
    mean = mean + bias

    variance = (var_term_0/N 
            + np.exp(-omega*t)*(var_term_1-var_term_0)/N 
            + np.exp(-2*omega*t)*(var_init/np.power(N,2) - var_term_1/N)
        )
    return mean, variance

def get_conf_int(amdata, t=np.linspace(0,100, 1_00)):

    mean, variance = bio_model_stats(amdata, t)

    a = ((1-mean)/variance - 1/mean)*np.power(mean,2)
    b = a*(1/mean - 1)

    conf_int = np.array(beta.interval(0.95, a, b))
    low_conf = conf_int[0]
    upper_conf = conf_int[1]

    return mean, low_conf, upper_conf

def bio_model_plot (amdata, alpha=1, fits=None, ax=None, hue='grey'):
    """Plot the evolution of site predicted by bio_model"""
    xlim=(0,100)
    t = np.linspace(xlim[0],xlim[1], 1_000)

    mean, variance = bio_model_stats(amdata, t)

    k = (mean*(1-mean)/variance)-1
    a = mean*k
    b = (1-mean)*k

    # a = ((1-mean)/variance - 1/mean)*np.power(mean,2)
    # b = a*(1/mean - 1)

    conf_int = np.array(beta.interval(0.95, a, b))
    low_conf = conf_int[0]
    upper_conf = conf_int[1]

    if ax is None:
        ax = plot.row(amdata.obs.index[0])

    sns.scatterplot(x=amdata.var.age,
                    y=amdata.X.flatten(),
                    # hue=hue,
                    # label='data',
                    alpha=alpha, ax=ax
                    )

    if fits is not None:
        site = amdata.obs.index[0]
        mean_slope, mean_inter, var_inter = fits.xs((site, 'linear'), level=['site', 'model'])['mean'].values
        mean_y = mean_slope*np.array([xlim[0],xlim[1]]) + mean_inter
        std2_plus = mean_y+2*np.sqrt(var_inter)
        std2_minus = mean_y-2*np.sqrt(var_inter)

        sns.lineplot(x=[xlim[0],xlim[1]], y=mean_y, ax=ax, label='linear_mean', color='tab:orange')
        sns.lineplot(x=[xlim[0],xlim[1]], y=std2_plus, ax=ax, color='tab:orange', label='linear_2-std')
        sns.lineplot(x=[xlim[0],xlim[1]], y=std2_minus, ax=ax, color='tab:orange')
        
    sns.lineplot(x=t, y=mean, color='tab:blue', label='mean',ax=ax)
    sns.lineplot(x=t, y=low_conf, color='tab:cyan', label='2-std',ax=ax)
    sns.lineplot(x=t, y=upper_conf, color='tab:cyan',ax=ax)

    ax.set_ylabel('methylation')
    ax.set_xlabel('age')
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(0,1)

    ax.legend(title='Bio_model')


    return ax

def is_saturating(amdata):
    """Check if site is saturating at birth or 100yo.
    We use the predicted 95% CI from the bio_model"""

    t = np.array([0, 100])
    mean, variance = bio_model_stats(amdata, t)
    
    a = ((1-mean)/variance - 1/mean)*np.power(mean,2)
    b = a*(1/mean - 1)

    conf_int = np.array(beta.interval(0.95, a, b))
    low_conf = conf_int[0]
    upper_conf = conf_int[1]

    if np.min(low_conf) < 0.05:
        return True

    elif np.max(upper_conf)>0.95:
        return True

    else:
        return False

def is_saturating_vect(amdata):
    """Check if site is saturating at birth or 100yo.
    We use the predicted 95% CI from the bio_model"""

    t = np.array([99, 100])
    mean, variance = bio_model_stats_vect(amdata, t)
    
    a = ((1-mean)/variance - 1/mean)*np.power(mean,2)
    b = a*(1/mean - 1)

    conf_int = np.array(beta.interval(0.95, a, b))
    intervals = pd.DataFrame(np.concatenate(conf_int, axis=1), index=amdata.obs.index)
    
    intervals['saturating'] = False
    intervals.loc[intervals[[0,1]].min(axis=1)< 0.05, 'saturating'] = True
    intervals.loc[intervals[[2,3]].max(axis=1)>0.95, 'saturating'] = True
    
    return intervals['saturating']

def mean_abs_derivative_at_point(amdata, t=100):
    obs = amdata.obs
    return np.abs(-obs.omega*np.exp(-obs.omega*t)*(obs.meth_init-obs.eta_0))


def comparison_plot(comparisons, n_sites, scale=True):
    """Plot model comparison grouped by sites, using the unified comparison df."""
    if n_sites != -1:
        sites = comparisons.index.get_level_values(level='site').unique()[:n_sites]

        plot_df = comparisons.reset_index()[:2*n_sites].copy()

    else: 
        plot_df = comparisons.reset_index().copy()
        plot_df = plot_df.groupby('model').sum(numeric_only=True).reset_index()
        sites =['all_sites']

    # Create indexing for plot
    index_group = plot_df.index.values - plot_df.index.values % 2
    plot_df['plot_index'] = ((plot_df.model =='linear')*1 - 0.5)/2 + index_group
    plot_df['plot_index'] = -plot_df['plot_index']

    parameter_plot = 'elpd_loo'
    center = plot_df['elpd_loo'].mean()
    if n_sites != -1 and scale is True:
        means = plot_df.groupby('site', axis=0).transform('mean')
        plot_df['elpd_loo_centered'] = plot_df['elpd_loo'] - means['elpd_loo']
        parameter_plot = 'elpd_loo_centered'
        center = 0
    
    # Create errorbar plot
    fig = px.scatter(plot_df,
            x=parameter_plot,
            y='plot_index',
            error_x='se',
            color='model',
            template='simple_white')
    
    fig.add_vline(x=center, line=dict(color='dimgrey',
                                 width=2,
                                 dash='dash') )
    if n_sites != -1:
        fig.update_yaxes(title='Site')
        fig.update_layout(
            yaxis = dict(
                tickmode = 'array',
                tickvals = -plot_df.index.values*2,
                ticktext = sites)
            )
    else:
        fig.update_yaxes(title='')
        fig.update_layout(
            yaxis = dict(
                tickmode = 'array',
                tickvals = -plot_df.index.values*2,
                ticktext = sites)
            )

    return fig



def part_bootstrap_map (filtered_sites_amdata, part=False, show_progress=True, initialisations = 10, method='Nelder-Mead'):
    """Find map """

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    omega = np.broadcast_to(filtered_sites_amdata.obs.omega, shape=(filtered_sites_amdata.shape[1], filtered_sites_amdata.shape[0])).T
    eta_0 = np.broadcast_to(filtered_sites_amdata.obs.eta_0, shape=(filtered_sites_amdata.shape[1], filtered_sites_amdata.shape[0])).T
    p = np.broadcast_to(filtered_sites_amdata.obs.meth_init, shape=(filtered_sites_amdata.shape[1], filtered_sites_amdata.shape[0])).T
    var_init = np.broadcast_to(filtered_sites_amdata.obs.var_init, shape=(filtered_sites_amdata.shape[1], filtered_sites_amdata.shape[0])).T
    N = np.broadcast_to(filtered_sites_amdata.obs.system_size, shape=(filtered_sites_amdata.shape[1], filtered_sites_amdata.shape[0])).T
    site_params = [omega, eta_0, p, var_init, N]

    age = filtered_sites_amdata.var.age.values

    # Create a list of deterministic models 
    model_fit = []

    iterator = tqdm(range(initialisations)) if show_progress else range(initialisations)
    for  i in iterator:
        # Each mutation has their own origin. 
        # Upper bound depends on first non-zero observation

        bounds = [(-2,2), (-1,1)]

        # initialise parameters uniformly inside bounds
        x0 = [np.random.normal(0, 0.05, size=filtered_sites_amdata.n_participants)
            for param in bounds]

        # while not (((-2 < x0[0]) & (x0[0]<2)).all() and 
        #             ((-1 < x0[1]) & (x0[1]<1)).all()):
        #     # initialise parameters uniformly inside bounds
        #     x0 = [np.random.normal(0, 0.1)
        #         for param in bounds]      

        # find optimal origins of deterministic trajectories
        fit = minimize(acc_bias_map, x0=x0,
                    # bounds=bounds,
                    args=(filtered_sites_amdata, site_params, age),
                    method=method,
                    # options={'maxiter': 1000},
                    # tol=0.0000001
                    )

        model_fit.append(fit)

    # find deterministic model wth minimum squared error.
    optimal_map = min(model_fit, key=attrgetter('fun'))    

    if part is not False:
        return np.array([fit.x[part] for fit in model_fit]), optimal_map
    
    return optimal_map#dict(zip(['acc', 'bias'], optimal_map.x))


def acc_bias_map(params, filtered_sites_amdata, site_params, age):
    acc = params[0]
    bias = params[1]
    omega, eta_0, p, var_init, N = site_params

    # Useful variables
    omega = np.exp(acc)*omega
    eta_1 = 1-eta_0
    
    # model mean and variance
    var_term_0 = eta_0*eta_1
    var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)


    mean = eta_0 + np.exp(-omega*age)*((p-1)*eta_0 + p*eta_1) + bias

    variance = (var_term_0/N 
            + np.exp(-omega*age)*(var_term_1-var_term_0)/N 
            + np.exp(-2*omega*age)*(var_init/np.power(N,2) - var_term_1/N)
        )

    nll = -norm.logpdf(filtered_sites_amdata.X, loc= mean, scale=np.sqrt(variance)).sum()

    return nll

def concat_maps(results):
    return pd.concat([pd.DataFrame(result['map']) for result in results])


def site_offsets(amdata, normal=True, return_MAP=True, return_trace=False, show_progress=False,
                         map_method='L-BFGS-B'):

    # The data has two dimensions: participant and CpG site
    coords = {"site": amdata.obs.index, "part": amdata.var.index}

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    omega = np.broadcast_to(amdata.obs.omega, shape=(amdata.shape[1], amdata.shape[0]))
    eta_0 = np.broadcast_to(amdata.obs.eta_0, shape=(amdata.shape[1], amdata.shape[0]))
    p = np.broadcast_to(amdata.obs.meth_init, shape=(amdata.shape[1], amdata.shape[0]))
    var_init = np.broadcast_to(amdata.obs.var_init, shape=(amdata.shape[1], amdata.shape[0]))
    N = np.broadcast_to(amdata.obs.system_size, shape=(amdata.shape[1], amdata.shape[0]))

    ages = np.broadcast_to(amdata.var.age, shape=(amdata.n_obs, amdata.n_vars)).T
    

    # Define Pymc model
    with pm.Model(coords=coords) as model:
        
        # Define priors
        offset = pm.Uniform("offset",   lower=-1, upper=1, dims='site')

        # Useful variables
        # omega = np.exp(acc)*omega
        eta_1 = 1-eta_0
        
        # model mean and variance
        var_term_0 = eta_0*eta_1
        var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)


        mean = eta_0 + np.exp(-omega*ages)*((p-1)*eta_0 + p*eta_1) + offset

        variance = (var_term_0/N 
                + np.exp(-omega*ages)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*ages)*(var_init/np.power(N,2) - var_term_1/N)
            )

        if normal is True:
            # Define likelihood
            obs = pm.Normal("obs",
                             mu=mean,
                             sigma = np.sqrt(variance), 
                             dims=("part", "site"), 
                             observed=amdata.X.T)
       

        if normal is False:
            # Force mean and variance in acceptable range
            mean = pm.math.minimum(mean, 0.001)
            mean = pm.math.maximum(mean, 0.999)
            variance = pm.math.minimum(variance, mean*(1-mean))
            variance = pm.math.maximum(variance, 0.001)

            # Define likelihood
            obs = pm.Beta("obs", mu=mean,
                                 sigma = np.sqrt(variance), 
                                 dims=("part", "site"),     
                                 observed=amdata.X.T)

        res = {}
        if return_MAP:
            res['map'] = pm.find_MAP(progressbar=show_progress, method=map_method, maxeval=10_000)

    return res    

