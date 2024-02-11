################################
# this script requires scripts below 
# site_modelling.py 
# downsampling.py 

# %% 
# IMPORTING

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import modelling_bio_beta as model

DATASET_NAME = 'wave3'
N_PART = 1_0

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_sites_fitted.h5ad',backed='r')
amdata = amdata[~amdata.obs.saturating].to_memory()
amdata = amdata[amdata.obs.sort_values('spr2', ascending=False).index]

accs = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_accs.csv', index_col=0)
biases = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_biases.csv', index_col=0)

n_sites_grid = [2**n for n in range(1,11)]
n_sites_label = [f'{n_sites_grid[i]}-{n_sites_grid[i+1]}' for i in range(len(n_sites_grid))[:-1]]

# %%
# Find worst spr2 in each downsample experiment
worst_sprs = []
for n_sites in n_sites_grid[1:]:
    worst_sprs.append(amdata[n_sites].obs.spr2.values[0])
# %% 
# PLOTTING
# Downsample acceleration
plot.fonts(8)
plt.rc('xtick', labelsize=6)

ax = plot.row(figsize=(9.5, 6))

df = pd.DataFrame(np.abs(np.diff(accs, axis=1)), index=accs.index, columns=n_sites_label)
sns.boxplot(ax=ax, data=df, color=colors[0], showfliers=False)
ax.set_xlabel('Change in number of sites in sets')
ax.set_ylabel('Absolute difference \nin acceleration')


ax2 = plt.twinx()
sns.lineplot(ax=ax2, x=n_sites_label, y=worst_sprs, color=colors[1], label='Worst site in set')
sns.scatterplot(ax=ax2, x=n_sites_label, y=worst_sprs, color=colors[1])
ax2.set_ylabel('spearman^2')

sns.despine(right=False)
ax.get_figure().tight_layout()

# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3B1_boxplot_site_downsampling.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3B1_boxplot_site_downsampling.svg')

# %% 
# PLOTTING

# Downsample bias

plot.fonts(8)
plt.rc('xtick', labelsize=6)

ax = plot.row(figsize=(9.5, 6))



df = pd.DataFrame(np.abs(np.diff(biases, axis=1)), index=biases.index, columns=n_sites_label)
sns.boxplot(df, color=colors[0], showfliers=False, ax=ax)
ax.set_xlabel('Change in number of sites in sets')
ax.set_ylabel('Absolute difference \nin bias')

ax2 = plt.twinx()
sns.lineplot(ax=ax2, x=n_sites_label, y=worst_sprs, color=colors[1], label='Worst site in set')
sns.scatterplot(ax=ax2, x=n_sites_label, y=worst_sprs, color=colors[1])
ax2.set_ylabel('spearman^2')

sns.despine(right=False)
ax.get_figure().tight_layout()

# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3B2_boxplot_site_downsampling.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3B2_boxplot_site_downsampling.svg')

# %%
