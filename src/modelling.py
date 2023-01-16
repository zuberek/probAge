import pymc as pm
import numpy as np
import arviz as az
import pandas as pd

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


def accelerated_biased_person_model(age, m_values):

    with pm.Model():

        # Define priors
        bias = pm.Uniform("bias", lower=-1, upper=1)
        acc = pm.Uniform("acc", lower=-2, upper=2)

        # Define likelihood
        likelihood = pm.Normal("m-values",
            mu=np.exp(acc)*slope_maps*age + inter_maps + bias,
            sigma=np.sqrt(drift_maps*age) + var_maps,
            observed=m_values)

        trace = pm.sample(cores=1, progressbar=False)
        map = pm.find_MAP(progressbar=False)

    return trace, map


