'''
Apply the model on a external dataset
'''

# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
from src import preprocess_func

# %%
# LOAD

EXTERNAL_OPEN_PATH = '../exports/wave1_meta.h5ad'
EXTERNAL_SAVE_PATH = '../exports/wave1_fitted.h5ad'
REFERENCE_DATA_PATH = '../exports/ewas_fitted.h5ad'

amdata = ad.read_h5ad(EXTERNAL_OPEN_PATH, backed='r')
amdata_ref = ad.read_h5ad(REFERENCE_DATA_PATH)

amdata_ref = amdata_ref[~amdata_ref.obs.saturating]

# Load intersection of sites in new dataset
params = modelling_bio.get_site_params()

# intersection = site_info.index.intersection(amdata.obs.index)
intersection = amdata_ref.obs.index.intersection(amdata.obs.index)
amdata = amdata[intersection].to_memory()

amdata.obs[params + ['r2']] = amdata_ref.obs[params + ['r2']]
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index]
amdata = amdata.copy()

# %%
# BATCH CORRECTION

# Infer the offsets
maps = modelling_bio.site_offsets(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata.obs['offset'] = maps['offset']
sns.histplot(amdata.obs.offset, bins=50)
# amdata = amdata[amdata.obs.sort_values('offset').index].copy()

# apply the offsets
amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset


# %%
# OPTIONAL DOWNSAMPLING EXPERIMENT

# n_sites_grid = [2**n for n in range(1,11)]
# n_sites_label = [f'{n_sites_grid[i]}-{n_sites_grid[i+1]}' for i in range(len(n_sites_grid))[:-1]]
# accs = np.empty((len(n_sites_grid), amdata.shape[1]))
# biases = np.empty((len(n_sites_grid), amdata.shape[1]))
# for i, n_sites in enumerate(tqdm(n_sites_grid)):
#     map = modelling_bio.person_model(amdata[:n_sites], return_MAP=True, return_trace=False)['map']
#     accs[i] = map['acc']
#     biases[i] = map['bias']    
# accs = pd.DataFrame(accs, index=n_sites_grid, columns=amdata.var.index).T
# biases = pd.DataFrame(biases, index=n_sites_grid, columns=amdata.var.index).T

# df = pd.DataFrame(np.abs(np.diff(accs, axis=1)), index=amdata.var.index, columns=n_sites_label)
# ax=sns.boxplot(df, color=colors[0], showfliers=False)
# ax.set_ylabel('Absolute difference in acceleration')

# df = pd.DataFrame(np.abs(np.diff(biases, axis=1)), index=amdata.var.index, columns=n_sites_label)
# ax=sns.boxplot(df, color=colors[0], showfliers=False)
# ax.set_ylabel('Absolute difference in bias')

# %%
# PERSON MODELLING

ab_maps = modelling_bio.person_model(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata.var['acc'] = ab_maps['acc']
amdata.var['bias'] = ab_maps['bias']

amdata.var.to_csv('wave1_ewas_var.csv')

amdata.write_h5ad(EXTERNAL_SAVE_PATH)