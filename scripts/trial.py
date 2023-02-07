# %%

%load_ext autoreload
%autoreload 2
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
import arviz as az

import pickle

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

# Load bio sites
# amdata = amdata_src.AnnMethylData('../exports/wave3_bio.h5ad')
amdata = amdata_src.AnnMethylData('../exports/hannum_acc_bio.h5ad')

amdata = amdata[amdata.obs.sort_values('r2', ascending=False)]
sns.scatterplot(x=amdata.var.age, y=amdata[].X.flatten())

N_PARTS = 100
N_CORES = 7

# drop saturating sites
amdata = amdata[amdata.obs['saturating'] == False]

# Fitting acceleration and bias
print('Acceleration and bias model fitting')
if N_PARTS is not False:
    amdata = amdata[:250, :N_PARTS]

# create amdata chunks to vectorize acc and bias over participants
chunk_size = 10
n_participants = amdata.shape[1]
amdata_chunks = []
for i in range(0, n_participants, chunk_size):
    amdata_chunks.append(amdata[:,i:i+chunk_size])

results = []
for chunk in tqdm(amdata_chunks):
    results.append(
        modelling_bio.person_model(chunk
        )
    )

traces = results[0]['trace']
modelling_bio.make_clean_trace(traces)
if len(results) > 1:
    for trace in tqdm(results[1:]):
        modelling_bio.concat_traces(traces, trace['trace'], dim='part')


map = modelling_bio.person_model(amdata[:, :], show_progress=True, return_trace=False, return_MAP=True)

map['map']['acc'] = np.log2(map['map']['acc'])
sns.scatterplot(x=amdata.var.acc_mean, y=map['map']['acc'])

sns.histplot(map['map']['acc'])


to_save = ['mean', 'sd', 'hdi_3%', 'hdi_97%']
to_save_acc = [f'acc_{col}' for col in to_save]
to_save_bias = [f'bias_{col}' for col in to_save]
amdata.var[to_save_acc] = az.summary(traces, var_names=['acc'])[to_save].values
amdata.var[to_save_bias] = az.summary(traces, var_names=['bias'])[to_save].values

amdata.var[to_save_acc] = np.log2(amdata.var[to_save_acc])
# drift_model = pd.read_csv('../exports/wave3_participants.csv')
sns.jointplot(amdata.var, x='acc_mean', y='bias_mean')