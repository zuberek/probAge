%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths
from scipy import stats
from sklearn.impute import KNNImputer



from src import modelling_bio

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

N_CORES = 15
R2_THRESHOLD = 0.3
TOP_SITES = None # number of sites to take in for the final person inferring


site_nan_sum = np.isnan(amdata.X).sum(axis=1)
empty_sites = site_nan_sum < (amdata.shape[1]*0.1)
amdata = amdata[empty_sites]

sample_nan_sum = np.isnan(amdata.X).sum(axis=0)
empty_samples = sample_nan_sum < (amdata.shape[0]*0.1)
amdata = amdata[:, empty_samples]

y = np.array(pd.Series(empty_sites).value_counts())
mylabels = ["Empty sites", "Full sites"]
plt.pie(y, labels = mylabels, startangle = 90)


 
###
# Impute missing values

# Sort values by age for KNN imputer
amdata = amdata[:, amdata.var.sort_values(by='age').index]

top_missing_site = amdata.obs.assign(nan_count=np.isnan(amdata.X).sum(axis=1)).sort_values('nan_count').tail(1).index[0]
site_data = amdata[top_missing_site]
nan_idx = np.argwhere(np.isnan(amdata.X))
imp = KNNImputer(n_neighbors=25)
amdata.X = imp.fit_transform(amdata.X.T).T
amdata.obs['r2'] = r_regression(amdata.X.T, amdata.var.age)**2


amdata = amdata_src.AnnMethylData('../exports/ewasKNN_fitted.h5ad')
amdata.obs['spr2'] = amdata.obs.spr**2
amdata = amdata[amdata.obs.sort_values('spr2').tail(5000).index]
amdata = amdata.to_memory()
amdata.obs['spr'].hist()

sns.scatterplot(data=amdata.obs, x='r2', y='spr2')

set1 = set(amdata[amdata.obs.sort_values('spr2').tail(1000).index].obs.index)
set2 = set(amdata[amdata.obs.sort_values('r2').tail(1000).index].obs.index)
venn2([set1, set2], ('spr2', 'r2'))


#####################################
### MODELLING SITES

# chunk_size = amdata.shape[0]//(N_CORES)
chunk_size = 11
n_sites = amdata.shape[0]
amdata_chunks = []
for i in range(0, n_sites, chunk_size):
    amdata_chunks.append(amdata[i:i+chunk_size])



import pymc as pm
amdata_chunks[0].shape
model = modelling_bio.bio_model(amdata_chunks[0])
pm.model_to_graphviz(model)
amdata_chunks = amdata_chunks[:-1]
len(amdata_chunks)
amdata_chunks[-1]

model = modelling_bio.bio_model(amdata_chunks[0])
# %%
%%timeit
res = modelling_bio.bio_model_fit(model,amdata_chunks[0], return_MAP=True, return_trace=False, show_progress=True)


# %%
%%timeit
modelling_bio.bio_sites(amdata_chunks[0], return_MAP=True, return_trace=True, show_progress=True)
# %%
%%timeit
with model:
    # Switch out the observed dataset
    pm.set_data({"data": amdata[0:13].X.T})
    pm.find_MAP(progressbar=True)


res = list(map(partial(modelling_bio.bio_model_fit, model, return_trace=False, return_MAP=True, show_progress=True), amdata_chunks))

# %%
with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=partial(modelling_bio.bio_model_fit, 
                            model, 
                            return_MAP = True,
                            return_trace = False,
                            show_progress=False),
                iterable=amdata_chunks,
                chunksize=1
                ), 
            total=len(amdata_chunks)))
  # %%

 

# no multiprocessing option
site_maps= modelling_bio.bio_sites(amdata[:10], return_MAP=True, return_trace=False, show_progress=True)['map']

site_maps = modelling_bio.concat_maps(results)
for param in modelling_bio.get_site_params():
    amdata.obs[param] = site_maps[param].values










##############################
idata  = modelling_bio.bio_sites(amdata['cg16938363'], return_MAP=False, return_trace=True, show_progress=True)['trace']
maps = modelling_bio.bio_sites(amdata['cg16938363'], return_MAP=True, return_trace=False, show_progress=True)['map']
idata = modelling_bio.bio_sites(amdata['cg22454769'], return_MAP=False, return_trace=True, show_progress=True)['trace']


import arviz as az
params = modelling_bio.get_site_params()
maps


amdata['cg22454769'].write_h5ad('bad_site2.h5ad')

pd.Series(maps)[params].values.ravel()

type(maps)
az.plot_posterior(idata, var_names=params)
az.plot_posterior(idata_norm, var_names=params)

site_data = amdata['cg16938363'].copy()

site_data.obs[params] = az.summary(idata).iloc[-5:,:]['mean'].values
site_data.obs[params] = maps[params].values.flatten()



amdata.obs.omega.sort_values()
amdata[]


modelling_bio.bio_model_plot(site_data)
modelling_bio.bio_model_plot(amdata['cg22454769'])
amdata['cg16938363'].obs[params]



az.plot_trace(idata)
amdata['cg22454769'].obs
az.summary(idata)
















amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(amdata, t=90)
amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(amdata)
amdata.obs['saturating_der'] = amdata.obs.abs_der<0.0005

amdata.obs['saturating'] = amdata.obs.saturating_std | amdata.obs.saturating_der



amdata.obs.saturating.value_counts()

import matplotlib.pyplot as plt

amdata[amdata.obs.saturating_std]
amdata.obs.saturating_std.index


sns.scatterplot(data=amdata.obs, x='spr2', y='omega')

from matplotlib_venn import venn2

set1 = set(amdata[amdata.obs.saturating_std].obs.index)
set2 = set(amdata[amdata.obs.saturating_der].obs.index)
venn2([set1, set2], ('saturating_std', 'saturating_der'))


amdata.write_h5ad(paths.DATA_FITTED) 
amdata
amdata.write_h5ad('../exports/ewasKNN2_fitted.h5ad') 

#####################################
### MODELLING PEOPLE
amdata = amdata_src.AnnMethylData(paths.DATA_FITTED)
amdata = amdata_src.AnnMethylData('../exports/ewasKNN2_fitted.h5ad')
amdata.obs.dtypes

amdata = amdata[~amdata.obs.saturating]
# amdata = amdata[amdata.obs.sort_values('r2').tail(250).index].copy()
amdata = amdata[amdata.obs.sort_values('r2').tail(500).index].copy()
amdata.obs.meth_init
person_maps = modelling_bio.person_model(
                            amdata[:,:100], 
                            return_trace=False, return_MAP=True, show_progress=True)['map']


data = amdata[:,:1000].copy()

amdata.var['acc'] = person_maps['acc']
amdata.var['bias'] = person_maps['bias']
data.var['acc'] = person_maps['acc']
data.var['bias'] = person_maps['bias']


sns.scatterplot(data=data.var, x='acc', y='bias', hue='age')

amdata.var.acc

sns.scatterplot(data=amdata.var, x='acc', y='acc_full')



amdata.write_h5ad(paths.DATA_FITTED)