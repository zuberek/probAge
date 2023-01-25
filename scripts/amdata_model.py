# %load_ext autoreload 
# %autoreload 2
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling
import arviz as az

import pickle

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

N_SITES =  1024

amdata = amdata.AnnMethylData('../exports/wave1_linear.h5ad')
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index[:N_SITES]]

# sns.scatterplot(x= amdata.var.age, y=amdata['cg07547549'].X.flatten())

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling.fit_and_compare,
                iterable=amdata, 
                ), 
            total=amdata.n_obs))

def multiprocess(function, data, total):
    with Pool(15, maxtasksperchild=1) as p:
        results = list(tqdm(
                iterable= p.imap(
                    func=function,
                    iterable=data, 
                    ), 
                total=total))
    return results
    
res = modelling.fit_and_compare(amdata[10])
res = multiprocess(modelling.drift_sites, amdata[:10], 10)

linear_trace2 = linear_trace.copy()
# compressed_posterior = linear_trace.posterior.mean(dim='chain').expand_dims(dim='chain', axis=0)
avg_likelihood = linear_trace.log_likelihood.mean(dim='chain').mean(dim='draw')
setattr(linear_trace2, 'log_likelihood', avg_likelihood)
az.loo(linear_trace2)
az.plot_trace(linear_trace2)
az.plot_posterior(linear_trace2)
az.summary(linear_trace2)

linear_trace = modelling.linear_sites(amdata[:10], show_progress=True)['trace']
drift_trace = modelling.drift_sites(amdata[:10], show_progress=True)['trace']

linear_trace, linear_map = linear_trace
drift_trace, drift_map = drift_trace

modelling.make_clean_trace(linear_trace)
comparison = az.compare({"drift": drift_trace, "linear": linear_trace})

amdata.write_h5ad('../exports/wave3_linear.h5ad')


chunk_size = 10
amdata_chunks = []
for i in range(0, 150, chunk_size):
    amdata_chunks.append(amdata[:,i:i+chunk_size])

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling.vector_person_model,
                iterable=amdata_chunks, 
                ), 
            total=len(amdata_chunks)))

traces = results[0]
modelling.make_clean_trace(traces)
for trace in tqdm(results[1:]):
    modelling.concat_traces(traces, trace, dim='part')


# traces.to_netcdf('../exports/participant_traces.nc', compress=False)
with open('../exports/linear_traces.pk', 'wb') as f:
    pickle.dump(linear_trace, f)


with open('../exports/trace.pk', 'rb') as f:
    traces = pickle.load(f, encoding='bytes')
    
part_idx = '202915470056_R04C01'
trace = traces.sel(part=part_idx)
az.plot_trace(traces.sel(part=part_idx))
az.plot_pair(traces, coords={'part': part_idx}, kind=['hexbin', 'kde'], point_estimate='mean', marginals=True)
az.summary(trace)

sns.histplot(traces.posterior.acc.values.mean(axis=0).mean(axis=0))
