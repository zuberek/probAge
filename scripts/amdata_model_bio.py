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

N_SITES = slice(10)
N_PARTS = slice(None)
N_CORES = 7

OPEN_PATH = '../exports/ewas_linear.h5ad'
SAVE_PATH = '../exports/ewas_model_comparison.h5ad'



amdata = amdata_src.AnnMethylData(OPEN_PATH, backed='r')
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index[N_SITES]].to_memory()
amdata = amdata_src.AnnMethylData(amdata)

with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=partial(modelling_bio.fit_and_compare, show_progress=True),
                iterable=amdata,
                chunksize=1
                ), 
            total=amdata.n_obs))

print('Exporting site model results')
fits, comparisons = modelling_bio.comparison_postprocess(results, amdata)
amdata.write_h5ad(SAVE_PATH)
# comparisons.to_csv('../exports/comparison_bio.csv')
# fits.to_csv('../exports/fits_bio.csv')

full_comparisons_plot = (
    modelling_bio.full_comparison_plot(comparisons))
# full_comparisons_plot.figure.savefig('../results/full_comparison.png')

# amdata.write_h5ad('../exports/wave3_bio.h5ad')

# drop saturating sites
amdata = amdata[amdata.obs['saturating'] == False]

# Fitting acceleration and bias
print('Acceleration and bias model fitting')
amdata = amdata[:, N_PARTS]

person_results = modelling_bio.person_model(amdata, show_progress=True)

amdata.var['acc'] = person_results['acc']
amdata.var['bias'] = person_results['bias']


# amdata.write_h5ad('../exports/wave3_acc_bio.h5ad')

# with open('../exports/bio_traces.pk', 'wb') as f:
#     pickle.dump(traces, f)




# # create amdata chunks to vectorize acc and bias over participants
# chunk_size = 10
# n_participants = amdata.shape[1]
# amdata_chunks = []
# for i in range(0, n_participants, chunk_size):
#     amdata_chunks.append(amdata[:,i:i+chunk_size])

# results = []
# for chunk in tqdm(amdata_chunks):
#     results.append(
#         modelling_bio.person_model(chunk
#         )
#     )

# traces = results[0]['trace']
# modelling_bio.make_clean_trace(traces)
# if len(results) > 1:
#     for trace in tqdm(results[1:]):
#         modelling_bio.concat_traces(traces, trace['trace'], dim='part')

# to_save = ['mean', 'sd', 'hdi_3%', 'hdi_97%']
# to_save_acc = [f'acc_{col}' for col in to_save]
# to_save_bias = [f'bias_{col}' for col in to_save]
# amdata.var[to_save_acc] = az.summary(traces, var_names=['acc'])[to_save].values
# amdata.var[to_save_bias] = az.summary(traces, var_names=['bias'])[to_save].values
# amdata.var[to_save_acc] = np.log2(amdata.var[to_save_acc])