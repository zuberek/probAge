%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths



from src import modelling_bio

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

N_CORES = 15
R2_THRESHOLD = 0.3
TOP_SITES = None # number of sites to take in for the final person inferring


amdata = amdata_src.AnnMethylData(paths.DATA_PROCESSED)
amdata.obs['spr2'] = amdata.obs.spr**2
amdata = amdata[amdata.obs.sort_values('spr2').tail(5000).index]
amdata = amdata.to_memory()
amdata.obs['spr'].hist()

sns.scatterplot(data=amdata.obs, x='r2', y='spr2')


#####################################
### MODELLING SITES

# chunk_size = amdata.shape[0]//(N_CORES)
chunk_size = 11
n_sites = amdata.shape[0]
amdata_chunks = []
for i in range(0, n_sites, chunk_size):
    amdata_chunks.append(amdata[i:i+chunk_size])

with Pool(N_CORES, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=partial(modelling_bio.bio_sites, 
                            return_MAP=True, 
                            return_trace=False, 
                            show_progress=False),
                iterable=amdata_chunks,
                chunksize=1
                ), 
            total=len(amdata_chunks)))

# no multiprocessing option
site_maps= modelling_bio.bio_sites(amdata[:10], return_MAP=True, return_trace=False, show_progress=True)['map']

site_maps = modelling_bio.concat_maps(results)
for param in modelling_bio.get_site_params():
    amdata.obs[param] = site_maps[param].values


amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(amdata, t=90)
amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(amdata)
amdata.obs['saturating_der'] = amdata.obs.abs_der<0.0005



amdata.obs['saturating'] = amdata.obs.saturating_std | amdata.obs.saturating_der


amdata.write_h5ad(paths.DATA_FITTED) 

#####################################
### MODELLING PEOPLE
amdata = amdata_src.AnnMethylData(paths.DATA_FITTED)
amdata.obs.dtypes

amdata = amdata[~amdata.obs.saturating]
# amdata = amdata[amdata.obs.sort_values('r2').tail(250).index].copy()
amdata = amdata[amdata.obs.sort_values('r2').tail(500).index].copy()
amdata.obs.meth_init
person_maps = modelling_bio.person_model(
                            amdata, 
                            return_trace=False, return_MAP=True, show_progress=True)['map']

amdata.var['acc'] = person_maps['acc']
amdata.var['bias'] = person_maps['bias']

amdata.write_h5ad(paths.DATA_FITTED)