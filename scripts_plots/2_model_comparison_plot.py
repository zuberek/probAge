%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import modelling_bio_beta as modelling_bio

amdata = ad.read_h5ad('../exports/wave3_acc.h5ad')
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index]

# Load site fits
fits_bio = pd.read_csv('../exports/fits_bio.csv', index_col=[0,1])
comparison = pd.read_csv('../exports/comparison_bio.csv', index_col=[0,1])

modelling_bio.bio_model_plot(amdata[0], alpha=0.3, fits=fits_bio, xlim=(0,125))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig('../results/Modelling/site_comparison.svg')


comparison =  comparison.loc[amdata.obs.index, slice(None)]

# LOO comparison
loo_site_plot = (
    modelling_bio.comparison_plot(comparison, n_sites=20))

loo_site_plot.write_image('../results/Modelling/loo_sites.svg', height=500, width=500 )


# LOO comparison
loo_full_plot = (
    modelling_bio.comparison_plot(comparison, n_sites=-1))

loo_full_plot.write_image('../results/Modelling/loo_full.svg', height=200, width=500 )

a = loo_full_plot.get_subplot

import plotly.graph_objs as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=1, row_heights=[0.1, 0.9])


fig.add_trace(loo_site_plot.data[0], row=2, col=1)
fig.add_trace(loo_site_plot.data[1], row=2, col=1)
