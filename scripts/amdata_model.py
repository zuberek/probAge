# %load_ext autoreload 
# %autoreload 2
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling
import arviz as az

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

N_SITES =  1024

amdata = amdata.AnnMethylData('../exports/wave3_linear.h5ad')
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index[:N_SITES]]

# sns.scatterplot(x= amdata.var.age, y=amdata['cg07547549'].X.flatten())

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling.fit_and_compare,
                iterable=amdata, 
                ), 
            total=amdata.n_obs))

modelling.comparison_postprocess(results, amdata)
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
for trace in tqdm(results[2:]):
    modelling.concat_traces(traces, trace, dim='part')

traces.to_netcdf('../exports/participant_traces.nc', compress=False)
import pickle
with open('../exports/trace.pk', 'wb') as f:
    pickle.dump(traces, f)


part_idx = '202915420061_R01C01'
az.plot_pair(traces, coords={'part': part_idx}, kind=['hexbin', 'kde'], point_estimate='mean', marginals=True)

az.plot_trace(traces)
