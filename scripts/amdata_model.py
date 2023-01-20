%load_ext autoreload
%autoreload 2
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
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index[:N_SITES]].to_memory()

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


data = [amdata[:,i] for i in amdata.participants.index]
chunk_size = 10
test = []

for i in range(0, amdata.n_participants, chunk_size):
    test.append(amdata[:,i:i+chunk_size])

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling.vector_person_model,
                iterable=test, 
                # chunksize=100
                ), 
            total=len(test)))

traces0 = modelling.vector_person_model(test[0])
traces1 = modelling.vector_person_model(test[1])

part_idx = amdata.participants.index[0]
traces0.sel(part=part_idx)



# extend
traces0.extend(traces1)
part_idx = test[0].participants.index[0]
traces[part_idx]

# concat
traces = az.concat(traces0, traces1, dim='part')

az.plot_trace(traces0)
az.plot_trace(traces1)
