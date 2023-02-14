# %% IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
from functools import partial
import arviz as az

import pickle

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

amdata_path = '../exports/hannum.h5ad'
site_info_path = '../exports/wave3_acc_sites.csv' 
wave3_path = '../exports/wave3_MAP_acc.h5ad'

wave3 = ad.read_h5ad(wave3_path)
site_info = pd.read_csv(site_info_path, index_col=0)
amdata = ad.read_h5ad(amdata_path)
params = modelling_bio.get_site_params()


# %% PREPARE DATA
intersection = site_info.index.intersection(amdata.obs.index)
amdata = amdata[intersection]

# Remove values outside out beta distribution
amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)

amdata_truth = amdata_src.AnnMethylData(amdata.copy())
amdata_wave3 = amdata_src.AnnMethylData(amdata.copy())

# import plotly.express as px
# px.scatter(x=amdata.var.age, y=amdata[-5].X.flatten(), hover_name=amdata.var.index)

# Add ground truth from wave3
amdata_wave3.obs[params + ['r2']] = site_info[params + ['r2']]

# %% 
# REFIT THE GROUND TRUTH
maps = modelling_bio.bio_sites(amdata_truth, return_MAP=True, return_trace=False, show_progress=True)['map']
for param in params:
    amdata_truth.obs[f'{param}'] = maps[param]

amdata_truth.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(amdata_truth, t=90)
amdata_truth.obs['saturating_std'] = modelling_bio.is_saturating_vect(amdata_truth)
amdata_truth.obs['saturating_der'] = amdata_truth.obs.abs_der<0.001

amdata_truth.obs['saturating'] = amdata_truth.obs.saturating_std | amdata_truth.obs.saturating_der

# axs = plot.tab(params, 'Hannum - Wave 3 comparison', ncols=2, row_size=5, col_size=8)
# for i, param in enumerate(params):
#     sns.scatterplot(x=amdata_truth.obs[param], y=amdata_wave3.obs[param], ax=axs[i])

amdata_truth = amdata_truth[~amdata_truth.obs.saturating].copy()

# %% 
# CALCULATE GROUND TRUTH PREDICTIONS
ab_maps = modelling_bio.person_model(amdata_truth, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata_truth.var['acc'] = ab_maps['acc']
amdata_truth.var['bias'] = ab_maps['bias']

# Calculate predictions without any correction
ab_maps = modelling_bio.person_model(amdata_wave3, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata_wave3.var['raw_acc'] = ab_maps['acc']
amdata_wave3.var['raw_bias'] = ab_maps['bias']
# sns.histplot(amdata_wave3.var, x='raw_acc', y='raw_bias', cbar=True, ax=plot.row('Before'))

# Infer the offsets
maps = modelling_bio.site_offsets(amdata_wave3, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata_wave3.obs['offset'] = maps['offset']
# sns.histplot(amdata_wave3.obs.offset)
amdata_wave3= amdata_wave3[amdata_wave3.obs.sort_values('offset').index]

# %% 
# Plot the data correction
ax = plot.row('Top shifted down')
site_index = amdata_wave3.obs.index[-1]
sns.scatterplot(x=wave3.var.age, y=wave3[site_index].X.flatten(), label='wave3',ax=ax)
sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten(), label='hannum', ax=ax)
sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten()-amdata_wave3[site_index].obs.offset.values[0], label='hannum (corr)', ax=ax)

ax = plot.row('Top shifted up')
site_index = amdata_wave3.obs.index[0]
sns.scatterplot(x=wave3.var.age, y=wave3[site_index].X.flatten(), label='wave3',ax=ax)
sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten(), label='hannum', ax=ax)
sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten()-amdata_wave3[site_index].obs.offset.values[0], label='hannum (corr)', ax=ax)

# %% 
# Apply the offset and refit acceleration and bias
offset = np.broadcast_to(amdata_wave3.obs.offset, shape=(amdata_wave3.shape[1], amdata_wave3.shape[0])).T
amdata_wave3.X = amdata_wave3.X - offset
ab_maps = modelling_bio.person_model(amdata_wave3, return_MAP=True, return_trace=False, show_progress=True)['map']

# %% 

amdata_wave3.var['corr_acc'] = ab_maps['acc']
amdata_wave3.var['corr_bias'] = ab_maps['bias']

sns.histplot(amdata_wave3.var, x='corr_acc', y='corr_bias', cbar=True, ax=plot.row('After'))

acc_df = pd.concat([amdata_wave3.var[['corr_acc', 'raw_acc']], amdata_truth.var['acc']], axis=1)
bias_df = pd.concat([amdata_wave3.var[['corr_bias', 'raw_bias']], amdata_truth.var['bias']], axis=1)


acc_df['corr_acc_diff'] = acc_df.corr_acc - acc_df.acc
acc_df['notcorr_acc_diff'] = acc_df.raw_acc - acc_df.acc
bias_df['corr_bias_diff'] = bias_df.corr_bias - bias_df.bias
bias_df['notcorr_bias_diff'] = bias_df.raw_bias - bias_df.bias
df=pd.concat((acc_df, bias_df), axis=1)

g = sns.JointGrid()
sns.scatterplot(data=df, x='corr_acc_diff', y='corr_bias_diff', label='Corrected', ax=g.ax_joint)
sns.scatterplot(data=df, x='notcorr_acc_diff', y='notcorr_bias_diff', label='Not Corrected', ax=g.ax_joint)
sns.kdeplot(data=df, x='corr_acc_diff', ax=g.ax_marg_x)
sns.kdeplot(data=df, x='notcorr_acc_diff', ax=g.ax_marg_x)
sns.kdeplot(data=df, y='corr_bias_diff', ax=g.ax_marg_y)
sns.kdeplot(data=df, y='notcorr_bias_diff', ax=g.ax_marg_y)
g.refline(y=0, x=0)

g.savefig('../results/3_jointplot_batch_correction_differences.svg')


df=acc_df.rename(columns={'corr_acc':'corr','raw_acc':'raw','acc':'true'}).melt(var_name='type', value_name='acc')
df['bias'] = bias_df.melt()['value']

sns.jointplot(df, x='acc', y='bias', hue='type')

sns.kdeplot(data=acc_df.melt(), x='value', hue='variable')

sns.scatterplot(acc_df, alpha=0.1, x='acc', y='corr_acc', label='Corrected')
sns.scatterplot(acc_df, alpha=0.1, x='acc', y='raw_acc', label='Raw')
sns.lineplot(x=[-1,1],y=[-1,1])