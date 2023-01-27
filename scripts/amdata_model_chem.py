# %%
%load_ext autoreload 
%autoreload 2
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_chem
import arviz as az

import pickle

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

N_SITES =  7

amdata = amdata_src.AnnMethylData('../exports/wave3_linear.h5ad')
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index[:N_SITES]]

# %%
# sns.scatterplot(x= amdata.var.age, y=amdata['cg07547549'].X.flatten())
with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling_chem.fit_and_compare,
                iterable=amdata,
                chunksize=1
                ), 
            total=amdata.n_obs))

modelling_chem.comparison_postprocess(results, amdata)
#amdata.write_h5ad('../exports/wave3_chem.h5ad')


# %%
chunk_size = 10
amdata_chunks = []
for i in range(0, 150, chunk_size):
    amdata_chunks.append(amdata[:,i:i+chunk_size])

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling_chem.person_model,
                iterable=amdata_chunks, 
                ), 
            total=len(amdata_chunks)))

traces = results[0]
modelling_chem.make_clean_trace(traces)
for trace in tqdm(results[1:]):
    modelling_chem.concat_traces(traces, trace, dim='part')


# traces.to_netcdf('../exports/participant_traces.nc', compress=False)
with open('../exports/linear_traces.pk', 'wb') as f:
    pickle.dump(traces, f)


# %%

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_chem
import arviz as az

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

N_SITES =  8      

amdata = amdata.AnnMethylData('../exports/wave3_linear.h5ad')
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index[:N_SITES]]

# sns.scatterplot(x= amdata.var.age, y=amdata['cg07547549'].X.flatten())

with Pool(maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling_chem.fit_and_compare,
                iterable=amdata, 
                ), 
            total=amdata.n_obs))

#modelling.comparison_postprocess(results, amdata)

# %%
fits, comparisons = modelling_chem.comparison_postprocess(results, amdata)
# %%
comparisons.loc[(slice(None),'linear'), :].sum()
# %%
index = 4 
params = amdata[index].obs[['nu_0', 'nu_1', 'init_meth', 'var_init', 'system_size']].values[0]

def chemical_plot (nu_0, nu_1, p, var_init, N, site=None):
  omega = nu_0 + nu_1
  eta_0 = nu_0/omega
  eta_1 = nu_1/omega

  t= np.linspace(0,100, 1_000)
  mean = eta_0 + np.exp(-omega*t)*((p-1)*eta_0 + p*eta_1)

  var_term_0 = eta_0*eta_1
  var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)


  var = (var_term_0/N 
          + np.exp(-omega*t)*(var_term_1-var_term_0)/N 
          + np.exp(-2*omega*t)*(var_init/np.power(N,2) - var_term_1/N)
        )

  sns.lineplot(x=t, y=mean, color='orange')
  sns.lineplot(x=t, y=mean+2*np.sqrt(var), color='orange')
  sns.lineplot(x=t, y=mean-2*np.sqrt(var), color='orange')

  if site != None:
    sns.scatterplot(x=amdata[site].participants.age, y=amdata[site].X.flatten())

chemical_plot(*params, site=index)
plt.ylim(0,1)

# %%

import pymc as pm
def person_model(amdata, return_trace=True, return_MAP=False, show_progress=False):

    # The data has two dimensions: participant and CpG site
    coords = {"site": amdata.sites.index, "part": amdata.participants.index}

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    nu_0 = np.broadcast_to(amdata.sites.nu_0, shape=(amdata.shape[1], amdata.shape[0])).T
    nu_1 = np.broadcast_to(amdata.sites.nu_1, shape=(amdata.shape[1], amdata.shape[0])).T
    p = np.broadcast_to(amdata.sites.init_meth, shape=(amdata.shape[1], amdata.shape[0])).T
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

output = person_model(amdata[:, :5], show_progress=True)
# %%

az.plot_trace(output['trace'],
              combined=True,  
              #coords={'part': amdata.participants.index[0]}
              )