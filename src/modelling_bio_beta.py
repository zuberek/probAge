# general packages
import numpy as np
import pandas as pd
from tqdm import tqdm

# numerical and bayesian inference packages
import pymc as pm
import arviz as az
from scipy.stats import beta

# plotting packages
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from src.utils import plot

# Set global parameters

SITE_PARAMETERS = {
    'eta_0':    'eta_0',
    'omega':    'omega',
    'p':        'meth_init',
    'N':        'system_size',
    'var_init': 'var_init'
}

PARAMS = ['eta_0','omega','p','N','var_init']

def sample_to_uniform_age(amdata, n_part):
    """Sample participants from amdata uniformly between
    min and max age in the cohort"""

    if amdata.shape[1]>n_part:

        remaining_idx = list(amdata.var.index)
        used_idx = []
        for i in tqdm(range(n_part)):
            sampled_age = np.random.uniform(amdata.var.age.min(), amdata.var.age.max())
            near_idx = np.abs(amdata[:, remaining_idx].var.age 
                    - sampled_age).argmin()
            used_idx.append(remaining_idx[near_idx])
            remaining_idx.pop(near_idx)

        return used_idx
    
    else:
        return amdata.var.index

def make_chunks(amdata, chunk_size):
    n_sites = amdata.shape[0]
    amdata_chunks = []
    for i in range(0, n_sites, chunk_size):
        amdata_chunks.append(amdata[i:i+chunk_size].copy())
    return amdata_chunks

# MODEL
def bio_sites(data):

    ages = np.broadcast_to(data.var.age, shape=(data.shape[0], data.shape[1])).T
    coords = {'sites': data.obs.index.values,
            'participants': data.var.index.values}

    early_idx = data[:, data.var.age<np.quantile(data.var.age, 0.05)].var.index
    p_mean = data[:, early_idx].X.mean(axis=1)
    p_std = data[:, early_idx].X.std(axis=1)

    late_idx = data[:, data.var.age>np.quantile(data.var.age, 0.95)].var.index
    eta_mean = data[:, late_idx].X.mean(axis=1)
    eta_std = data[:, late_idx].X.std(axis=1)

    with pm.Model(coords=coords) as model:

        data = pm.MutableData("data", data.X.T)
        ages = pm.ConstantData("ages", ages)

        # Define priors
        p = pm.Beta("meth_init", mu=p_mean, sigma=p_std, dims='sites')
        eta_0 = pm.Beta("eta_0", mu=eta_mean, sigma=eta_std, dims='sites')
        omega = pm.Gamma("omega", mu = 0.001, sigma= 0.01, dims='sites')
        N = pm.Gamma('system_size', mu=100, sigma=100, dims='sites')
        prop_std_init = pm.Gamma("prop_std_init",
                                 mu=0.1, sigma=0.1, dims='sites')
        
        var_init = pm.Deterministic('var_init',
                                    prop_std_init*np.power(N, 2))

        # Useful variables
        eta_1 = 1-eta_0
        
        # model mean and variance
        var_term_0 = eta_0*eta_1
        var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

        mean = eta_0 + np.exp(-omega*ages)*(p-eta_0)

        variance = (var_term_0/N 
                + np.exp(-omega*ages)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*ages)*(var_init/np.power(N,2) - var_term_1/N)
            )

        # Beta function parameter boundaries
        sigma = np.sqrt(variance)
        sigma = pm.math.minimum(sigma,  pm.math.sqrt(mean * (1 - mean))-0.001)

        # Define likelihood
        observations = pm.Beta("m-values",
            mu = mean,
            sigma = sigma,
            dims=("participants", "sites"),
            observed = data)
        
    return model

def site_MAP(data_chunk, progressbar=False):
    """Create a function for  parallel inference of MAP
    given a data_chunk"""

    model = bio_sites(data_chunk)
    with model:
        map = pm.find_MAP(progressbar=progressbar)
    
    return map

def is_saturating_sd(amdata, t1=0, t2=100):
    """Check if site is saturating at birth or 100yo.
    We use the predicted 95% CI from the bio_model"""

    t = np.array([t1, t2])
    mean, variance = bio_model_stats_vect(amdata, t)
    
    a = ((1-mean)/variance - 1/mean)*np.power(mean,2)
    b = a*(1/mean - 1)

    conf_int = np.array(beta.interval(0.90, a, b))
    intervals = pd.DataFrame(np.concatenate(conf_int, axis=1), index=amdata.obs.index)
    
    intervals['saturating'] = False
    intervals.loc[intervals[[0,1]].min(axis=1)< 0.025, 'saturating'] = True
    intervals.loc[intervals[[2,3]].max(axis=1)> 0.975, 'saturating'] = True
    
    return intervals['saturating']

def mean_abs_derivative_at_point(amdata, t=100):
    obs = amdata.obs
    return np.abs(-obs.omega*np.exp(-obs.omega*t)*(obs.meth_init-obs.eta_0))

def get_saturation_inplace(amdata, early_age=0, old_age=100):
    """Calculate saturation based on standard deviation and 
    absolute derivative at two points in time"""
    amdata.obs['saturating_sd'] = is_saturating_sd(amdata, t1=early_age, t2=old_age)
    abs_der = mean_abs_derivative_at_point(amdata, t=old_age)
    amdata.obs['saturating_der'] = abs_der<0.0005
    amdata.obs.loc[:,'saturating'] = amdata.obs.saturating_sd | amdata.obs.saturating_der
    return amdata

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

# PLOTTING
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

def bio_model_plot (amdata, bio_fit=True, xlim=(0,100), alpha=1, fits=None, ax=None, hue='grey'):
    """Plot the evolution of site predicted by bio_model"""

    if xlim is None:
        xlim = (amdata.var.age.min(), amdata.var.age.max())

    if ax is None:
        ax = plot.row(amdata.obs.index[0])

    sns.scatterplot(x=amdata.var.age,
                    y=amdata.X.flatten(),
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
   
    if bio_fit is True:
        t = np.linspace(xlim[0],xlim[1], 1_000)

        mean, variance = bio_model_stats(amdata, t)

        k = (mean*(1-mean)/variance)-1
        a = mean*k
        b = (1-mean)*k

        conf_int = np.array(beta.interval(0.95, a, b))
        low_conf = conf_int[0]
        upper_conf = conf_int[1]    
            
        sns.lineplot(x=t, y=mean, color='tab:blue', label='mean',ax=ax)
        sns.lineplot(x=t, y=low_conf, color='tab:cyan', label='2-std',ax=ax)
        sns.lineplot(x=t, y=upper_conf, color='tab:cyan',ax=ax)


    ax.set_ylabel('methylation')
    ax.set_xlabel('age')
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(0,1)

    ax.legend(title='Bio_model')


    return ax
############################
### PERSON INFERENCE
def person_model(amdata, method='map', progressbar=False, map_method='Powell'):

    # The data has two dimensions: participant and CpG site
    coords = {"site": amdata.obs.index, "part": amdata.var.index}

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    omega = np.broadcast_to(amdata.obs.omega, shape=(amdata.shape[1], amdata.shape[0])).T
    eta_0 = np.broadcast_to(amdata.obs.eta_0, shape=(amdata.shape[1], amdata.shape[0])).T
    p = np.broadcast_to(amdata.obs.meth_init, shape=(amdata.shape[1], amdata.shape[0])).T
    var_init = np.broadcast_to(amdata.obs.var_init, shape=(amdata.shape[1], amdata.shape[0])).T
    N = np.broadcast_to(amdata.obs.system_size, shape=(amdata.shape[1], amdata.shape[0])).T

    # Define Pymc model
    with pm.Model(coords=coords) as model:
        
        # Define model variables
        acc = pm.Normal('acc', mu=0, sigma=1, dims='part')
        bias = pm.Normal('bias', mu=0, sigma=0.2, dims='part')

        # acc = pm.Normal('acc', mu=0, sigma=0.5, dims='part')
        # bias = pm.Normal('bias', mu=0, sigma=0.05, dims='part')

        data = pm.MutableData("data", amdata.X)
        age = pm.MutableData("age", amdata.var.age)

        # Useful variables
        omega = np.power(2, acc)*omega
        eta_1 = 1-eta_0
        
        # model mean and variance
        var_term_0 = eta_0*eta_1
        var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

        # Define mean
        mean = eta_0 + np.exp(-omega*age)*((p-1)*eta_0 + p*eta_1) + bias
        mean = pm.math.clip(mean, 0.001, 0.999)

        # Define variance
        variance = (var_term_0/N 
                + np.exp(-omega*age)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*age)*(var_init/np.power(N,2) - var_term_1/N)
            )
        
        # Beta function parameter boundaries
        variance = pm.math.clip(variance, 0, mean*(1-mean))
        sigma = np.sqrt(variance)

        # # Define likelihood
        obs = pm.Beta("obs",
                    mu=mean,
                    sigma = sigma, 
                    dims=("site", "part"), 
                    observed=data)
        
        # return model
        
        if method == 'map':
            return pm.find_MAP(method=map_method, progressbar=progressbar)

        if method == 'nuts':
            return pm.sample(1000, tune=1000, progressbar=progressbar)
        
def person_model_ll(amdata, acc_name='acc', bias_name='bias'):

    age = np.broadcast_to(amdata.var.age, shape=(amdata.shape))
    data = amdata.X

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    omega = np.broadcast_to(amdata.obs.omega, shape=(amdata.shape[1], amdata.shape[0])).T
    eta_0 = np.broadcast_to(amdata.obs.eta_0, shape=(amdata.shape[1], amdata.shape[0])).T
    p = np.broadcast_to(amdata.obs.meth_init, shape=(amdata.shape[1], amdata.shape[0])).T
    var_init = np.broadcast_to(amdata.obs.var_init, shape=(amdata.shape[1], amdata.shape[0])).T
    N = np.broadcast_to(amdata.obs.system_size, shape=(amdata.shape[1], amdata.shape[0])).T

    acc = np.array(amdata.var[acc_name])
    bias = np.array(amdata.var[bias_name])

    # Useful variables
    omega = np.power(2, acc)*omega
    eta_1 = 1-eta_0
    
    # model mean and variance
    var_term_0 = eta_0*eta_1
    var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

    # Define mean
    mean = eta_0 + np.exp(-omega*age)*((p-1)*eta_0 + p*eta_1) + bias
    mean = np.clip(mean, 0.001, 0.999)

    # Define variance
    variance = (var_term_0/N 
            + np.exp(-omega*age)*(var_term_1-var_term_0)/N 
            + np.exp(-2*omega*age)*(var_init/np.power(N,2) - var_term_1/N)
        )
    
    # Beta function parameter boundaries
    variance = np.clip(variance, 0, mean*(1-mean))

    k = (mean*(1-mean)/variance)-1
    a = mean*k
    b = (1-mean)*k

    return beta.logpdf(data, a=a, b=b).sum(axis=0)

def get_person_fit_quality(ab_ll, quantile=0.023):
    return ab_ll < ab_ll.quantile(quantile)

        
        

def site_offsets(amdata, show_progress=False, map_method='L-BFGS-B'):
# def site_offsets(amdata, show_progress=False, map_method='Powell'):
    
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
        offset = pm.Normal("offset",  mu=0, sigma=0.01, dims='site')

        # Useful variables
        eta_1 = 1-eta_0
        
        # model mean and variance
        var_term_0 = eta_0*eta_1
        var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

        mean = eta_0 + np.exp(-omega*ages)*(p-eta_0) + offset

        variance = (var_term_0/N 
                + np.exp(-omega*ages)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*ages)*(var_init/np.power(N,2) - var_term_1/N)
            )

        # Force mean and variance in acceptable range
        mean = pm.math.clip(mean, 0.001, 0.999)
        variance = pm.math.clip(variance, 0, mean*(1-mean))

        # Define likelihood
        obs = pm.Normal("obs", mu=mean,
                            #   sigma = 0.1, 
                                sigma = np.sqrt(variance), 
                                dims=("part", "site"),     
                                observed=amdata.X.T)

        return pm.find_MAP(progressbar=show_progress, method=map_method, maxeval=10_000)