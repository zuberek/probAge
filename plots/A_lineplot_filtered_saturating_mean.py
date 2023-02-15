#%%
# IMPORTS
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
import matplotlib.patches as mpatches

amdata = ad.read_h5ad('../exports/wave3_all_fitted.h5ad')
increasing = amdata.obs.eta_0>amdata.obs.meth_init

#%%
# DATA PREP
xlim=(0,100)
t = np.linspace(xlim[0],xlim[1], 1_00)

omegas = np.broadcast_to(amdata.obs.omega, shape=(xlim[1], amdata.shape[0])).T
etas = np.broadcast_to(amdata.obs.eta_0, shape=(xlim[1], amdata.shape[0])).T
ps = np.broadcast_to(amdata.obs.meth_init, shape=(xlim[1], amdata.shape[0])).T

means = modelling_bio.bio_site_mean(t, etas, omegas, ps)

legend_handles = [
    mpatches.Patch(color=colors[1], label='Saturating Mean'),
    mpatches.Patch(color='#F39B7FFF', label='Saturating Derivative'),
    mpatches.Patch(color=colors[0], label='Not Saturating'),
]

#%%
# PLOTTING ONE

# Saturating increasing sites
ax = plot.row('Saturating increasing sites')
for i, mean in enumerate(means[increasing]):
    color=colors[0]
    if amdata.obs[increasing].iloc[i].saturating_der: color = '#F39B7FFF'
    if amdata.obs[increasing].iloc[i].saturating_std: color = colors[1] 
    sns.lineplot(x=t, y=mean, color=color, alpha=0.2, ax=ax)

ax.legend(handles=legend_handles, loc='right')
ax.set_ylim((0,1))
ax.axhline(y=0.05, color=colors[1], linestyle='dotted')
ax.axhline(y=0.95, color=colors[1], linestyle='dotted')

plot.save(ax, 'A_lineplot_filtered_increasing_saturating_means', format='svg')
plot.save(ax, 'A_lineplot_filtered_increasing_saturating_means', format='png')

#%%
# PLOTTING TWO

# Saturating decreasing sites
ax = plot.row('Saturating decreasing sites')
for i, mean in enumerate(means[~increasing]):
    color=colors[0]
    if amdata.obs[~increasing].iloc[i].saturating_der: color = '#F39B7FFF'
    if amdata.obs[~increasing].iloc[i].saturating_std: color = colors[1] 
    sns.lineplot(x=t, y=mean, color=color, alpha=0.2, ax=ax)

handles = [
    mpatches.Patch(color=colors[1], label='Saturating Mean'),
    mpatches.Patch(color='tab:orange', label='Saturating Derivative'),
    mpatches.Patch(color=colors[0], label='Not Saturating'),
]
ax.legend(handles=legend_handles, loc='right')
ax.set_ylim((0,1))
ax.axhline(y=0.05, color='tab:red', linestyle='dotted')
ax.axhline(y=0.95, color='tab:red', linestyle='dotted')

plot.save(ax, 'A_lineplot_filtered_decreasing_saturating_means', format='svg')
plot.save(ax, 'A_lineplot_filtered_decreasing_saturating_means', format='png')

# %%
