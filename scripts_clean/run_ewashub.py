%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.feature_selection import r_regression

from src import modelling_bio

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

N_CORES = 15


amdata = amdata_src.AnnMethylData(paths.DATA_PROCESSED)
amdata = amdata[:,amdata.var.tissue=='whole blood']
# amdata = amdata[amdata[amdata.obs.r2>R2_THRESHOLD].obs.index].to_memory()
# amdata = amdata_src.AnnMethylData(amdata.copy())

# remove sites and participants with too many missing values
site_nan_sum = np.isnan(amdata.X).sum(axis=1)
amdata = amdata[site_nan_sum < (amdata.shape[1]*0.1)]
sample_nan_sum = np.isnan(amdata.X).sum(axis=0)
amdata = amdata[:, sample_nan_sum < (amdata.shape[0]*0.1)]
 
# Sort values by age for KNN imputer
amdata = amdata[:, amdata.var.sort_values(by='age').index]

#####################################
### PROCESSING

chunk_size = amdata.shape[0]//(N_CORES*100)
n_sites = amdata.shape[0]
amdata_chunks = []
for i in range(0, n_sites, chunk_size):
    amdata_chunks.append(amdata[i:i+chunk_size])

def impute(sites):
    sites.X = KNNImputer(n_neighbors=25).fit_transform(sites.X.T).T

for chunk in tqdm(amdata_chunks):
    impute(chunk)

# Compute spearman correlation
def spearman_r_loop(site_idx):
    spr = stats.spearmanr(amdata[site_idx].X.flatten(), amdata.var.age)
    return spr.statistic

with Pool(15) as pool:
    result = list(tqdm(pool.imap(spearman_r_loop, amdata.obs.index), total=amdata.shape[0]))

amdata.obs['spr'] = result

# Compute pearson
amdata.obs['r2'] = r_regression(amdata.X.T, amdata.var.age)**2

amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)
amdata.var.loc[amdata.var.age==0,'age']=0.00001

amdata.write_h5ad(paths.DATA_PROCESSED)

#####################################
### MODELLING SITES

amdata = amdata_src.AnnMethylData(paths.DATA_PROCESSED)

sns.histplot(x=amdata.obs.spr**2)
amdata = amdata[amdata.obs.spr**2>0.2]
sns.histplot(x=amdata.obs.spr)

chunk_size = amdata.shape[0]//(N_CORES*10)
n_sites = amdata.shape[0]
amdata_chunks = []
for i in range(0, n_sites, chunk_size):
    amdata_chunks.append(amdata[i:i+chunk_size])
amdata_chunks[0]

  

amdata.var[amdata.var.age==0]
np.isnan(amdata.X).sum()
# no multiprocessing option
for chunk in tqdm(amdata_chunks):
    modelling_bio.bio_sites(chunk, return_MAP=True, return_trace=False, show_progress=True)['map']

site_maps= modelling_bio.bio_sites(amdata_chunks[0], return_MAP=True, return_trace=False, show_progress=True)['map']

site_maps = modelling_bio.concat_maps(results)
for param in modelling_bio.get_site_params():
    amdata.obs[param] = site_maps[param].values

amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(amdata, t=90)
amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(amdata)
amdata.obs['saturating_der'] = amdata.obs.abs_der<0.001

amdata.obs['saturating'] = amdata.obs.saturating_std | amdata.obs.saturating_der

# store in the same location since we're only adding data
amdata.write_h5ad(paths.DATA_PROCESSED) 

#####################################
### MODELLING PEOPLE

amdata = amdata[~amdata.obs.saturating].copy()
# amdata = amdata[amdata.obs.sort_values('r2').tail(250).index].copy()
amdata = amdata[amdata.obs.sort_values('r2').tail(250).index].copy()

person_maps = modelling_bio.person_model(
                            amdata, 
                            return_trace=False, return_MAP=True, show_progress=True)['map']

amdata.var['acc'] = person_maps['acc']
amdata.var['bias'] = person_maps['bias']


amdata.write_h5ad(paths.DATA_FITTED)