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
from src import paths

from src import modelling_bio

# %%
# LOAD

external = 'wave4'
reference = 'EWAS'

amdata = ad.read_h5ad(f'../exports/{external}_fitted.h5ad', backed='r')
participants = pd.read_csv(f'../exports/{external}_participants.csv')
amdata_ref = ad.read_h5ad(f'../exports/{reference}_fitted.h5ad', backed='r')
sites_ref = amdata_ref[~amdata_ref.obs.saturating].obs

# Load intersection of sites in new dataset
params = modelling_bio.get_site_params()

# intersection = site_info.index.intersection(amdata.obs.index)
intersection = sites_ref.index.intersection(amdata.obs.index)
intersection = sites_ref.loc[intersection].sort_values('spr2').tail(500).index

amdata = amdata[intersection].to_memory()
amdata.obs[params + ['r2', 'spr2']] = sites_ref[params + ['r2', 'spr2']]


# %% #################
# BATCH (MODEL) CORRECTION

if 'status' in amdata.var.columns:
    maps = modelling_bio.site_offsets(amdata[:,amdata.var.status=='control'], return_MAP=True, return_trace=False, show_progress=True)['map']
else:
    maps = modelling_bio.site_offsets(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
# Infer the offsets
amdata.obs['offset'] = maps['offset']
sns.histplot(amdata.obs.offset, bins=50)
# amdata = amdata[amdata.obs.sort_values('offset').index].copy()

# apply the offsets
amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset

# show the offset applied to data
# site_index = amdata.obs.offset.abs().sort_values().index[-1]
# sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten())
# sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten()-amdata[site_index].obs.offset.values)
# sns.scatterplot(x=amdata_ref.var.age, y=amdata_ref[site_index].X.flatten())


# %% ##################################
# OPTIONAL DOWNSAMPLING EXPERIMENT

# n_sites_grid = [2**n for n in range(1,11)]
# n_sites_label = [f'{n_sites_grid[i]}-{n_sites_grid[i+1]}' for i in range(len(n_sites_grid))[:-1]]
# accs = np.empty((len(n_sites_grid), amdata.shape[1]))
# biases = np.empty((len(n_sites_grid), amdata.shape[1]))
# for i, n_sites in enumerate(tqdm(n_sites_grid)):
#     map = modelling_bio.person_model(amdata[:n_sites], method='map', show_progress=True)
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

# %% ##################
# PERSON MODELLING  

ab_maps = modelling_bio.person_model(amdata, method='map', show_progress=True)

participants[f'acc_{reference}'] = ab_maps['acc']
participants[f'bias_{reference}'] = ab_maps['bias']

participants.to_csv(f'../exports/{external}_participants.csv')