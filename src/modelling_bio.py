import pymc as pm
import arviz as az

import numpy as np
from scipy.stats import norm, beta
# from pymc.sampling_jax import sample_numpyro_nuts
from pymc.sampling import jax
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import xarray as xr

# import color palette
import sys
sys.path.append("..") 
from src.general_imports import sns_colors

CHAINS = 4
CORES = 1
# cores are set at 1 to allow to 
# avoid daemonic children in mp

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
            res['trace'] = jax.sample_numpyro_nuts(1000, tune=1000, chains=CHAINS, chain_method='sequential', postprocessing_backend='cpu',  progressbar=show_progress) 
            # res['trace'] = pm.sample(1000, tune=1000, chains=CHAINS, cores=CORES, progressbar=show_progress) 
            pm.compute_log_likelihood(res['trace'], progressbar=show_progress)

    return res


def bio_sites(amdata, return_MAP=False, return_trace=True, show_progress=False, target_accept=0.9):

    ages = np.broadcast_to(amdata.participants.age, shape=(amdata.n_sites, amdata.n_participants)).T
    coords = {'sites': amdata.sites.index.values,
            'participants': amdata.participants.index.values}

    with pm.Model(coords=coords) as model:
        init_std_bound = 0.1
        # Define priors
        nu_0 = pm.Uniform("nu_0", lower=0, upper=0.1, dims='sites')
        nu_1 = pm.Uniform("nu_1", lower=0, upper=0.1 , dims='sites')
        p = pm.Uniform("meth_init", lower=0, upper=1, dims='sites')
        N = pm.Uniform('system_size', lower= 1, upper=100_000, dims='sites')
        var_init = pm.Uniform("var_init", lower=0,
                                          upper=np.power(init_std_bound*N,2),
                                          dims='sites')

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
            # res['trace'] = pm.sample(1000, tune=1000,
            #                          chains=CHAINS, cores=CORES,
            #                          progressbar=show_progress,
            #                          target_accept=target_accept) 
            res['trace'] = jax.sample_numpyro_nuts(1000, tune=1000, chains=CHAINS, chain_method='sequential', postprocessing_backend='cpu',  progressbar=show_progress) 

            pm.compute_log_likelihood(res['trace'], progressbar=show_progress)

    return res

def bio_sites_reparam(amdata, return_MAP=False, return_trace=True, show_progress=False, init_nuts='auto', target_accept=0.9):

    ages = np.broadcast_to(amdata.participants.age, shape=(amdata.n_sites, amdata.n_participants)).T
    coords = {'sites': amdata.sites.index.values,
            'participants': amdata.participants.index.values}

    with pm.Model(coords=coords) as model:
        # condition on maximal initial standard deviation
        init_std_bound = 0.1

        # Define priors
        eta_0 = pm.Uniform("eta_0", lower=0, upper=1, dims='sites')
        omega = pm.Uniform("omega", lower=0, upper=1, dims='sites')
        p = pm.Uniform("meth_init", lower=0, upper=1, dims='sites')
        N = pm.Uniform('system_size', lower= 1, upper=100_000, dims='sites')
        var_init = pm.Uniform("var_init", lower=0,
                                          upper=np.power(init_std_bound*N,2),
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
            # res['trace'] = pm.sample(1000, tune=1000, init=init_nuts,
            #                          chains=CHAINS, cores=CORES,
            #                          progressbar=show_progress,
            #                          target_accept=target_accept)
            res['trace'] = jax.sample_numpyro_nuts(1000, tune=1000,
                                chains=CHAINS, chain_method='sequential',
                                postprocessing_backend='cpu', 
                                progressbar=show_progress)
 
            pm.compute_log_likelihood(res['trace'], progressbar=show_progress)

    return res

def fit_and_compare(amdata, show_progress=False):

    ROUND = 7

    trace_l = linear_sites(amdata, show_progress=show_progress)['trace']
    trace_bio = bio_sites_reparam(amdata, show_progress=show_progress)['trace']

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

def bio_fit(amdata, show_progress=False):
    ROUND = 7
    trace_bio = bio_sites_reparam(amdata, show_progress=show_progress)['trace']
    bio_fit = az.summary(trace_bio, round_to=ROUND)
    bio_fit.index = pd.MultiIndex.from_tuples([(index_tuple[1][:-1], 'bio', index_tuple[0]) for index_tuple in bio_fit.index.str.split('[')],
                            names=['site', 'model', 'param'])

    return bio_fit

def bio_fit_post(results, amdata):
    fits = pd.DataFrame()
    for fit in results:
        fits = pd.concat([fits, fit])


    # Extract parameter names from bio model
    param_list = fit.xs('bio', level='model').index.get_level_values(level='param')
    for param in param_list:
        amdata.obs[param] = fits.loc[(slice(None),'bio', param)]['mean'].values

    return fits

def person_model(amdata, normal=True, return_trace=True, return_MAP=True, show_progress=False):

    # The data has two dimensions: participant and CpG site
    coords = {"site": amdata.obs.index, "part": amdata.var.index}

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    nu_0 = np.broadcast_to(amdata.obs.nu_0, shape=(amdata.shape[1], amdata.shape[0])).T
    nu_1 = np.broadcast_to(amdata.obs.nu_1, shape=(amdata.shape[1], amdata.shape[0])).T
    p = np.broadcast_to(amdata.obs.meth_init, shape=(amdata.shape[1], amdata.shape[0])).T
    var_init = np.broadcast_to(amdata.obs.var_init, shape=(amdata.shape[1], amdata.shape[0])).T
    N = np.broadcast_to(amdata.obs.system_size, shape=(amdata.shape[1], amdata.shape[0])).T

    age = amdata.var.age.values


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

        variance = (var_term_0/N 
                + np.exp(-omega*age)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*age)*(var_init/np.power(N,2) - var_term_1/N)
            )
        
        # Force mean and variance in acceptable range
        mean = pm.math.minimum(mean, 1)
        mean = pm.math.maximum(mean,0)
        variance = pm.math.minimum(variance, mean*(1-mean))
        variance = pm.math.maximum(variance, 0)

        if normal is True:
            # Define likelihood
            obs = pm.Normal("obs",
                             mu=mean,
                             sigma = np.sqrt(variance), 
                             dims=("site", "part"), 
                             observed=amdata.X)
       

        if normal is False:
            # Force mean and variance in acceptable range
            mean = pm.math.minimum(mean, 0.001)
            mean = pm.math.maximum(mean, 0.999)
            variance = pm.math.minimum(variance, mean*(1-mean))
            variance = pm.math.maximum(variance, 0.001)

            # Define likelihood
            obs = pm.Beta("obs", mu=mean,
                                 sigma = np.sqrt(variance), 
                                 dims=("site", "part"), 
                                 observed=amdata.X)

        res = {}
        if return_MAP:
            res['map'] = pm.find_MAP(progressbar=False)

        if return_trace:
            res['trace'] = jax.sample_numpyro_nuts(1000, tune=1000, chains=CHAINS, chain_method='sequential', postprocessing_backend='cpu',  progressbar=show_progress) 

    return res


def person_model_reparam(amdata, normal=True, return_trace=True, return_MAP=True, show_progress=False):

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
        acc = pm.Uniform('acc', lower=-2, upper = 2, dims='part')
        bias = pm.Uniform('bias', lower=-1, upper = 1, dims='part')

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

        if normal is True:
            # Define likelihood
            obs = pm.Normal("obs",
                             mu=mean,
                             sigma = np.sqrt(variance), 
                             dims=("site", "part"), 
                             observed=amdata.X)
       

        if normal is False:
            # Force mean and variance in acceptable range
            mean = pm.math.minimum(mean, 0.001)
            mean = pm.math.maximum(mean, 0.999)
            variance = pm.math.minimum(variance, mean*(1-mean))
            variance = pm.math.maximum(variance, 0.001)

            # Define likelihood
            obs = pm.Beta("obs", mu=mean,
                                 sigma = np.sqrt(variance), 
                                 dims=("site", "part"), 
                                 observed=amdata.X)

        res = {}
        if return_MAP:
            res['map'] = pm.find_MAP(progressbar=False)

        if return_trace:
            res['trace'] = jax.sample_numpyro_nuts(1000, tune=1000, chains=CHAINS, chain_method='sequential', postprocessing_backend='cpu',  progressbar=show_progress) 

    return res    

def make_clean_trace(trace):
    delattr(trace, 'sample_stats')
    delattr(trace, 'observed_data')

def concat_traces(trace1, trace2, dim):
    for group in ['posterior']:
        concatenated_group = xr.concat((trace1[group], trace2[group]), dim=dim)
        setattr(trace1, group, concatenated_group)


def bio_model_stats(amdata, t):
    """Extract mean and variace of site at a given set of 
    time-points."""

    # Extract parameters from site
    eta_0 = amdata.obs['eta_0'].to_numpy()
    omega = amdata.obs['omega'].to_numpy()
    var_init = amdata.obs['var_init'].to_numpy()
    p = amdata.obs['meth_init'].to_numpy()
    N = amdata.obs['system_size'].to_numpy()

    # reparametrization
    eta_1 = 1-eta_0

    # model mean and variance
    var_term_0 = eta_0*eta_1
    var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

    mean = eta_0 + np.exp(-omega*t)*((p-1)*eta_0 + p*eta_1)

    variance = (var_term_0/N 
            + np.exp(-omega*t)*(var_term_1-var_term_0)/N 
            + np.exp(-2*omega*t)*(var_init/np.power(N,2) - var_term_1/N)
        )
    return mean, variance

def bio_model_plot (amdata):
    """Plot the evolution of site predicted by bio_model"""
    t = np.linspace(0,100, 1_000)

    mean, variance = bio_model_stats(amdata, t)

    a = ((1-mean)/variance - 1/mean)*np.power(mean,2)
    b = a*(1/mean - 1)

    conf_int = np.array(beta.interval(0.95, a, b))
    low_conf = conf_int[0]
    upper_conf = conf_int[1]

    sns.scatterplot(x=amdata.participants.age,
                    y=amdata.X.flatten(),
                    color=sns_colors[0],
                    label='data'
                    )

    sns.lineplot(x=t, y=mean, color='red', label='mean')
    sns.lineplot(x=t, y=low_conf, color='orange', label='2-std')
    sns.lineplot(x=t, y=upper_conf, color='orange')

    plt.ylabel('methylation')
    plt.xlabel('age')
    plt.ylim(0,1)

    plt.legend(title='Bio_model')

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