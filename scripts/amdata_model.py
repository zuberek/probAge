%load_ext autoreload 
%autoreload 2
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


amdata = amdata_src.AnnMethylData('../exports/wave1_linear.h5ad')

# sns.scatterplot(x= amdata.var.age, y=amdata['cg07547549'].X.flatten())

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling.fit_and_compare,
                iterable=amdata[:10], 
                ), 
            total=amdata.n_obs))
    
# res = modelling.fit_and_compare(amdata[10], show_progress=True)

amdata.write_h5ad('../exports/wave3_linear.h5ad')


chunk_size = 10
n_participants = amdata.n_participants
amdata_chunks = []
for i in range(0, n_participants, chunk_size):
    amdata_chunks.append(amdata[:,i:i+chunk_size])

with Pool(maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling.person_model,
                iterable=amdata_chunks, 
                ), 
            total=len(amdata_chunks)))

traces = results[0]['trace']
modelling.make_clean_trace(traces)
if len(results) > 1:
    for trace in tqdm(results[1:]):
        modelling.concat_traces(traces, trace['trace'], dim='part')

to_save = ['mean', 'sd', 'hdi_3%', 'hdi_97%']
to_save_acc = [f'acc_{col}' for col in to_save]
to_save_bias = [f'bias_{col}' for col in to_save]
amdata.var[to_save_acc] = az.summary(traces, var_names=['acc'], round_to=5)[to_save].values
amdata.var[to_save_bias] = az.summary(traces, var_names=['bias'], round_to=5)[to_save].values

amdata.write_h5ad('../exports/wave3_linear.h5ad')

ax1,ax2 = plot.row(['Acceleration', 'Bias'], 'Wave 3 Prediction Confidence')
sns.histplot(data=amdata.var, x='acc_mean', ax=ax1, y='acc_sd', cbar=True)
sns.histplot(data=amdata.var, x='bias_mean', ax=ax2, y='bias_sd', cbar=True)

with open('../exports/drift_person_traces.pk', 'wb') as f:
    pickle.dump(traces, f)

with open('../exports/person_traces.pk', 'rb') as f:
    traces = pickle.load(f, encoding='bytes')
    
