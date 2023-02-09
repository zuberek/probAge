%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
from functools import partial

from sklearn.feature_selection import r_regression
import arviz as az

import pickle

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

external_path = '../exports/hannum.h5ad'
# external_path = '../exports/wave3_meta.h5ad' 
external_cohort_name = 'hannum'

bad_sites = ['cg18279094']
amdata = amdata_src.AnnMethylData(external_path)
site_indexes = amdata.obs.sort_values('r2', ascending=False).index[:10].tolist() + bad_sites
amdata = amdata[site_indexes]
amdata = amdata_src.AnnMethylData(amdata)

amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)
amdata.write_h5ad(external_path)
def r2(site_index):
    return r_regression(amdata[site_index].X.T, amdata.var.age)[0]**2


with Pool() as p:
    output = list(tqdm(p.imap(
            func=r2, 
            iterable=amdata.obs.index,
            chunksize=len(amdata.obs.index)//(cpu_count()*4)
            ),
        total=len(amdata.obs.index)))
amdata.obs['r2'] = output


params = ['omega', 'eta_0', 'meth_init', 'var_init', 'system_size']
maps = modelling_bio.bio_sites(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
for param in params:
    amdata.obs[f'{param}'] = maps[param]
ab_maps = modelling_bio.person_model(amdata, return_trace=False, return_MAP=True, show_progress=True)['map']
amdata.var['acc'] = ab_maps['acc']
amdata.var['bias'] = ab_maps['bias']

axs = plot.tab(amdata.sites.index, ncols=5)
for i, site_index in enumerate(amdata.sites.index):
    modelling_bio.bio_model_plot(amdata[site_index], ax=axs[i])


amdata.var.age = amdata.var.age-18
for param in params:
    amdata.obs[f'true_{param}'] = amdata.obs[param]

maps = modelling_bio.bio_sites(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
for param in params:
    amdata.obs[param] = maps[param]
ab_maps = modelling_bio.person_model(amdata, return_trace=False, return_MAP=True, show_progress=True)['map']
amdata.var['shift_acc'] = ab_maps['acc']
amdata.var['shift_bias'] = ab_maps['bias']


sns.scatterplot(amdata.var, x='acc', y='shift_acc', hue='age')
sns.scatterplot(amdata.var, x='bias', y='shift_bias')
amdata[:,amdata.var.age<18].var.shift_acc.hist()


axs = plot.tab(params, ncols=2)
for i, param in enumerate(params):
    sns.scatterplot(amdata.obs, x=f'shift_{param}', y=param, ax=axs[i])

ab_maps = modelling_bio.person_model(amdata, return_trace=False, return_MAP=True, show_progress=True)['map']



N_SITES = 2
N_PARTS = False
N_CORES = 7

# load selected sites for fitting
with open("../resources/selected_sites.json", "r") as f:
    sites = json.load(f)

# load external cohort
amdata = amdata_src.AnnMethylData(external_path, backed='r')
# filter sites
amdata = amdata[amdata.sites.index.isin(sites)].to_memory()
amdata = amdata_src.AnnMethylData(amdata)

amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)

if N_SITES is not False:
    amdata = amdata[:N_SITES]

with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling_bio.bio_fit,
                iterable=amdata,
                chunksize=1
                ), 
            total=amdata.n_obs))

fits = modelling_bio.bio_fit_post(results,  amdata)

fits.to_csv('../exports/fits_' + external_cohort_name + '.csv')

# Fitting acceleration and bias
print('Acceleration and bias model fitting')
if N_PARTS is not False:
    amdata = amdata[:, :N_PARTS]

# create amdata chunks to vectorize acc and bias over participants
chunk_size = 10
n_participants = amdata.shape[1]
amdata_chunks = []
for i in range(0, n_participants, chunk_size):
    amdata_chunks.append(amdata[:,i:i+chunk_size])

# Modify fitting function to return only MAP
person_model = partial(modelling_bio.person_model, return_MAP=True, return_trace=False)

results = []
for chunk in tqdm(amdata_chunks):
    results.append(
        person_model(chunk
        )
    )

# Append acc and bias to amdata object
acc = np.concatenate([r['map']['acc'] for r in results])
bias = np.concatenate([r['map']['bias'] for r in results])

amdata.var['acc_mean'] = acc
amdata.var['bias_mean'] = bias

# Export amdata object
amdata.write_h5ad('../exports/' + external_cohort_name +'_acc_bio.h5ad')