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


# %% #########################################
# PLOTTING
# Downsample acceleration
plot.fonts(8)
plt.rc('xtick', labelsize=6)

acc_ax, bias_ax = plot.row(subtitles=['Acceleration', 'Bias'], 
                           figsize=(9.5*2, 6))

# Acceleration

df = pd.DataFrame(np.abs(np.diff(accs, axis=1)), index=accs.index, columns=n_sites_label)
sns.boxplot(ax=acc_ax, data=df, color=colors[0], showfliers=False)
acc_ax.set_xlabel('Change in number of sites in sets')
acc_ax.set_ylabel('Absolute difference')

acc_ax2 = acc_ax.twinx()
sns.lineplot(ax=acc_ax2, x=n_sites_label, y=worst_sprs, color=colors[1], 
             linewidth=1, label='Worst site in set')
sns.scatterplot(ax=acc_ax2, x=n_sites_label, y=worst_sprs, color=colors[1])
# acc_ax2.set_ylabel(r'Site correlation with age ($\rho^2$)')
acc_ax2.set_yticks([])
acc_ax.tick_params(axis='x', labelrotation=30)

sns.despine(ax=acc_ax, top=True, right=False)
sns.despine(ax=acc_ax2, top=True, right=False)

#  Bias
df = pd.DataFrame(np.abs(np.diff(biases, axis=1)), index=biases.index, columns=n_sites_label)
sns.boxplot(df, color=colors[0], showfliers=False, ax=bias_ax)
bias_ax.set_xlabel('Change in number of sites in sets')
bias_ax.set_ylabel(None)
bias_ax.tick_params(axis='x', labelrotation=30)

bias_ax2 = bias_ax.twinx()
sns.lineplot(ax=bias_ax2, x=n_sites_label, y=worst_sprs, color=colors[1], label='Worst site in set')
sns.scatterplot(ax=bias_ax2, x=n_sites_label, y=worst_sprs, color=colors[1])
bias_ax2.set_ylabel(r'Site correlation with age ($\rho^2$)')

sns.despine(ax=bias_ax, top=True, right=False)
sns.despine(ax=bias_ax2, top=True, right=False)

# %%
acc_ax.get_figure().tight_layout()
acc_ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3B_boxplot_site_downsampling.png')
acc_ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3B_boxplot_site_downsampling.svg')
















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
