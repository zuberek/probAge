import numpy as np

import pymc as pm

import src.modelling_bio_beta as modelling

def merge_external(external, reference):
    # Load intersection of sites in new dataset
    params = list(modelling.SITE_PARAMETERS.values())

    # intersection = site_info.index.intersection(amdata.obs.index)
    intersection = reference.obs.index.intersection(external.obs.index)

    external = external[intersection].to_memory()
    external.obs[params] = reference.obs[params]

    return external

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
        offset = pm.Normal("offset",  mu=0, sigma=0.1, dims='site')

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

