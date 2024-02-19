# %% ########################
# LOADING

%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import plotly.express as px
import numpy as np
import pandas as pd
import pickle

amdata = ad.read_h5ad('../exports/wave3_person_fitted.h5ad')

with open('../exports/comparison.pickle', 'rb') as f:
    data = pickle.load(f)

data[0]

# %% ########################
# PROCESSING
n_sites=25
site_index = np.repeat(np.arange(len(data)),2)*2

# create plot positions array
plot_index = np.arange(len(data))*2
# reverse it so 
# plot_index = max(plot_index) - plot_index

site_name = amdata.obs.index[len(data):]
# site_name =[f'site{i}' for i in range(n_sites)]

df_comp = pd.concat(data).reset_index()
df_comp = df_comp
df_comp = df_comp.rename(columns={'index': 'model'})
df_comp['site'] = site_index

# center the comparisons
means = df_comp.groupby('site', axis=0)['elpd_loo'].transform('mean')
df_comp['elpd_loo_centered'] = df_comp['elpd_loo'] - means

# get positions of the grouped comparisons
df_comp['plot_index'] = -((df_comp.model =='lin')*1 - 1)/2 + site_index
df_comp['plot_index_reverse'] = -df_comp['plot_index']

# %% ########################
# PLOTTING 1
fig = px.scatter(df_comp[:2*n_sites], x='elpd_loo_centered', y='plot_index_reverse', color='model',
        error_x='se', template='simple_white', height=500, width=500,
        color_discrete_sequence=['rgb(77, 187, 213)', 'rgb(230, 75, 53)'])

fig.update_layout(
    yaxis = dict(
        tickmode = 'array',
        tickvals = -plot_index[:n_sites],
        ticktext = site_name[:n_sites])
    )


# %% ########################
# PLOTTING 2
df_agg = df_comp.groupby('model').sum().reset_index()
df_agg = df_agg.sort_values(by='elpd_loo', ascending=False)
full_fig = px.scatter(df_agg, x='elpd_loo', y =[0.25,-0.25], error_x='se', color='model', 
                      template='simple_white', colors=[colors[0], colors[1]], height=200, width=500)
full_fig.update_layout(
    yaxis = dict(
        tickmode = 'array',
        tickvals = [0],
        ticktext = ['All sites'])
    )
# %% saving
fig.write_image(f'{paths.FIGURES_DIR}/2CA_model_comparison.svg')
fig.write_image(f'{paths.FIGURES_DIR}/2CA_model_comparison.png')

full_fig.write_image(f'{paths.FIGURES_DIR}/2CB_model_comparison.svg')
full_fig.write_image(f'{paths.FIGURES_DIR}/2CB_model_comparison.png')