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

N_SITES =  10

amdata = amdata_src.AnnMethylData('../exports/wave3_linear.h5ad')
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index[:N_SITES]]

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling_chem.fit_and_compare,
                iterable=amdata,
                chunksize=1
                ), 
            total=amdata.n_obs))

fits, comparisons = modelling_chem.comparison_postprocess(results, amdata)
comparisons.to_csv('../exports/comparison_chem.csv')
fits.to_csv('../exports/fits_chem.csv')

# create amdata chunks to vectorize acc and bias over participants
chunk_size = 10
n_participants = amdata.shape[1]
amdata_chunks = []
for i in range(0, n_participants, chunk_size):
    amdata_chunks.append(amdata[:,i:i+chunk_size])

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling_chem.person_model,
                iterable=amdata_chunks, 
                ), 
            total=len(amdata_chunks)))

traces = results[0]['trace']
modelling_chem.make_clean_trace(traces)
if len(results) > 1:
    for trace in tqdm(results[1:]):
        modelling_chem.concat_traces(traces, trace['trace'], dim='part')

to_save = ['mean', 'sd', 'hdi_3%', 'hdi_97%']
to_save_acc = [f'acc_{col}' for col in to_save]
to_save_bias = [f'bias_{col}' for col in to_save]
amdata.var[to_save_acc] = az.summary(traces, var_names=['acc'])[to_save].values
amdata.var[to_save_bias] = az.summary(traces, var_names=['bias'])[to_save].values

amdata.write_h5ad('../exports/wave3_chem.h5ad')

with open('../exports/chem_traces.pk', 'wb') as f:
    pickle.dump(traces, f)

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

