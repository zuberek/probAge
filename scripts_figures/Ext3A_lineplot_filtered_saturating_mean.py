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



#%%
ax1, ax2 = plot.row(['Increasing sites', 'Decreasing sites'], 
                    figsize=(9.5*2, 6))
plot.fonts(8)
sns.despine()

legend_handles = [
    mpatches.Patch(color=colors[1], label='Saturating\nSD'),
    mpatches.Patch(color='#F39B7FFF', label='Saturating\nDerivative'),
    mpatches.Patch(color=colors[0], label='Not\nSaturating'),
]
#%%
# Saturating increasing sites
for i, mean in enumerate(tqdm(means[increasing])):
    color=colors[0]
    if amdata.obs[increasing].iloc[i].saturating_der: color = '#F39B7FFF'
    if amdata.obs[increasing].iloc[i].saturating_sd: color = colors[1] 
    sns.lineplot(x=t, y=mean, color=color, alpha=0.2, ax=ax1)

# ax1.legend(handles=legend_handles, loc='right')
ax1.set_ylim((0,1))
ax1.axhline(y=0.05, color=colors[1], linestyle='dotted')
ax1.axhline(y=0.95, color=colors[1], linestyle='dotted')

ax1.set_xlabel('Age (years)')
ax1.set_ylabel('Methylation level \n'+r'($\beta$-values)')

#%%
# Saturating decreasing sites

for i, mean in enumerate(tqdm(means[~increasing])):
    color=colors[0]
    if amdata.obs[~increasing].iloc[i].saturating_der: color = '#F39B7FFF'
    if amdata.obs[~increasing].iloc[i].saturating_sd: color = colors[1] 
    sns.lineplot(x=t, y=mean, color=color, alpha=0.2, ax=ax2)

# ax2.legend(handles=legend_handles, loc='right')
ax2.set_ylim((0,1))
ax2.axhline(y=0.05, color='tab:red', linestyle='dotted')
ax2.axhline(y=0.95, color='tab:red', linestyle='dotted')

ax2.set_xlabel('Age (years)')

legend = ax2.legend(handles=legend_handles, bbox_to_anchor=(1,1), loc='best')
# ax2.set_ylabel('Methylation level \n(beta values)')


#%%
ax1.get_figure().tight_layout()
ax1.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3A_lineplot_filtered_saturating_mean.png')
ax1.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3A_lineplot_filtered_saturating_mean.svg')

# %%
