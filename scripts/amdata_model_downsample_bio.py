#%%
# IMPORTS
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio

amdata = amdata_src.AnnMethylData('../exports/wave3_all_fitted.h5ad')
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index]
amdata = amdata[~amdata.obs.saturating].copy()
amdata = amdata_src.AnnMethylData(amdata)

n_sites_grid = [2**n for n in range(1,11)]
n_sites_label = [f'{n_sites_grid[i]}-{n_sites_grid[i+1]}' for i in range(len(n_sites_grid))[:-1]]

#%%
# COMPUTE DOWNSAMPLING ACC AND BIAS
accs = np.empty((len(n_sites_grid), amdata.n_participants))
biases = np.empty((len(n_sites_grid), amdata.n_participants))
for i, n_sites in enumerate(tqdm(n_sites_grid)):
    map = modelling_bio.person_model(amdata[:n_sites], return_MAP=True, return_trace=False)['map']
    accs[i] = map['acc']
    biases[i] = map['bias']    

#%%
# SAVE/LOAD DATA
accs = pd.DataFrame(accs, index=n_sites_grid, columns=amdata.participants.index).T
biases = pd.DataFrame(biases, index=n_sites_grid, columns=amdata.participants.index).T
accs.to_csv('../exports/downsample_accs.csv')
biases.to_csv('../exports/downsample_biases.csv')
accs = pd.read_csv('../exports/downsample_accs.csv', index_col=0)
biases = pd.read_csv('../exports/downsample_biases.csv', index_col=0)

r2_array = pd.Series([amdata.sites.iloc[i].r2 for i in n_sites_grid[1:]], name='smallest r2 value in iteration')

#%%
# PLOTTING ACCELERATION
plt.rc('font', size=8) 
plt.rc('xtick', labelsize=5) 
sns.despine(top=True, right=False)
ax = plot.row('', figsize=(3.6,2.6))
ax1 = ax.twinx()
df = pd.DataFrame(np.abs(np.diff(accs, axis=1)), index=amdata.participants.index, columns=n_sites_label)
sns.boxplot(df, color=colors[0], showfliers=False, ax=ax)
sns.lineplot(x=n_sites_label, y=r2_array, marker='o', color=colors[1], label='r2 value', ax=ax1)
ax.set_xlabel('Change in the number of methylation sites')
ax.set_ylabel('Absolute difference in acceleration')

plot.save(ax, 'A_acc_downsampling', format='svg')
plot.save(ax, 'A_acc_downsampling', format='png')

#%%
# PLOTTING BIAS

plt.rc('font', size=8) 
plt.rc('xtick', labelsize=5) 
ax = plot.row('', figsize=(3.6,2.6))
ax1 = ax.twinx()
df = pd.DataFrame(np.abs(np.diff(biases, axis=1)), index=amdata.participants.index, columns=n_sites_label)
sns.boxplot(df, color=colors[0], showfliers=False, ax=ax)
sns.lineplot(x=n_sites_label, y=r2_array, marker='o', color=colors[1], label='r2 value', ax=ax1)
ax.set_xlabel('Change in the number of methylation sites')
ax.set_ylabel('Absolute difference in bias')

sns.despine(top=True, right=False)
plot.save(ax, 'A_bias_downsampling', format='svg')
plot.save(ax, 'A_bias_downsampling', format='png')
# %%
