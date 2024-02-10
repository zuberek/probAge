#%%
# IMPORTS
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling_bio
import matplotlib.patches as mpatches

amdata = ad.read_h5ad('../exports/wave3_sites_fitted.h5ad')
increasing = amdata.obs.eta_0>amdata.obs.meth_init



#%%
# DATA PREP
xlim=(0,100)
t = np.linspace(xlim[0],xlim[1], 1_00)

omegas = np.broadcast_to(amdata.obs.omega, shape=(xlim[1], amdata.shape[0])).T
etas = np.broadcast_to(amdata.obs.eta_0, shape=(xlim[1], amdata.shape[0])).T
ps = np.broadcast_to(amdata.obs.meth_init, shape=(xlim[1], amdata.shape[0])).T

def bio_site_mean(ages, eta_0, omega, p):
    return eta_0 + np.exp(-omega*ages)*(p-eta_0)
means = bio_site_mean(t, etas, omegas, ps)

legend_handles = [
    mpatches.Patch(color=colors[1], label='Saturating SD'),
    mpatches.Patch(color='#F39B7FFF', label='Saturating Derivative'),
    mpatches.Patch(color=colors[0], label='Not Saturating'),
]

#%%
# PLOTTING ONE

plt.rc('font', size=8) 
ax = plot.row('', figsize=(3.6,2.6))
sns.despine()

# Saturating increasing sites
for i, mean in enumerate(means[increasing]):
    color=colors[0]
    if amdata.obs[increasing].iloc[i].saturating_der: color = '#F39B7FFF'
    if amdata.obs[increasing].iloc[i].saturating_sd: color = colors[1] 
    sns.lineplot(x=t, y=mean, color=color, alpha=0.2, ax=ax)

ax.legend(handles=legend_handles, loc='right')
ax.set_ylim((0,1))
ax.axhline(y=0.05, color=colors[1], linestyle='dotted')
ax.axhline(y=0.95, color=colors[1], linestyle='dotted')

ax.set_xlabel('Age (years)')
ax.set_ylabel('Methylation level (beta values)')
#%%()
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/Ext2A_lineplot_filtered_increasing_saturating_means.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/Ext2A_lineplot_filtered_increasing_saturating_means.svg')

#%%
# Saturating decreasing sites
plt.rc('font', size=8) 
ax = plot.row('', figsize=(3.6,2.6))
sns.despine()

for i, mean in enumerate(means[~increasing]):
    color=colors[0]
    if amdata.obs[~increasing].iloc[i].saturating_der: color = '#F39B7FFF'
    if amdata.obs[~increasing].iloc[i].saturating_sd: color = colors[1] 
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

ax.set_xlabel('Age (years)')
ax.set_ylabel('Methylation level (beta values)')

#%%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/Ext2A_lineplot_filtered_decreasing_saturating_means.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/Ext2A_lineplot_filtered_decreasing_saturating_means.svg')

# %%