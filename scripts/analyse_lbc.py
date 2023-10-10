# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
from src import preprocess_func

# %%
# LOAD

EXTERNAL_OPEN_PATH = '../exports/lbc2.h5ad'
EXTERNAL_SAVE_PATH = '../exports/lbc_fitted.h5ad'
REFERENCE_DATA_PATH = '../exports/ewas_fitted.h5ad'

amdata = ad.read_h5ad(EXTERNAL_OPEN_PATH)
amdata_ref = ad.read_h5ad(REFERENCE_DATA_PATH)

amdata_ref = amdata_ref[~amdata_ref.obs.saturating]

# Load intersection of sites in new dataset
params = modelling_bio.get_site_params()

# intersection = site_info.index.intersection(amdata.obs.index)
intersection = amdata_ref.obs.index.intersection(amdata.obs.index)
amdata = amdata[intersection].to_memory()

amdata.obs[params + ['r2']] = amdata_ref.obs[params + ['r2']]
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index]
amdata = amdata.copy()


amdata = amdata[:,amdata.var.reset_index().set_index('ID').loc[amdata.var.ID.value_counts()[amdata.var.ID.value_counts()>2].index].set_index('index').index]
amdata = amdata[:, amdata.var.cohort=='LBC36']
amdata = amdata[:, amdata.var[~np.isnan(amdata.var.age)].index]
amdata.write_h5ad('../exports/lbc.h5ad')

# %%
# BATCH CORRECTION

# Infer the offsets
maps = modelling_bio.site_offsets(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata.obs['offset'] = maps['offset']
sns.histplot(amdata.obs.offset, bins=50)
# amdata = amdata[amdata.obs.sort_values('offset').index].copy()

# apply the offsets
amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset

# %%
# Do the downsampling experiment
n_sites_grid = [2**n for n in range(1,11)]
n_sites_label = [f'{n_sites_grid[i]}-{n_sites_grid[i+1]}' for i in range(len(n_sites_grid))[:-1]]
accs = np.empty((len(n_sites_grid), amdata.shape[1]))
biases = np.empty((len(n_sites_grid), amdata.shape[1]))
for i, n_sites in enumerate(tqdm(n_sites_grid)):
    map = modelling_bio.person_model(amdata[:n_sites], return_MAP=True, return_trace=False)['map']
    accs[i] = map['acc']
    biases[i] = map['bias']    
accs = pd.DataFrame(accs, index=n_sites_grid, columns=amdata.var.index).T
biases = pd.DataFrame(biases, index=n_sites_grid, columns=amdata.var.index).T

df = pd.DataFrame(np.abs(np.diff(accs, axis=1)), index=amdata.var.index, columns=n_sites_label)
ax=sns.boxplot(df, color=colors[0], showfliers=False)
ax.set_ylabel('Absolute difference in acceleration')

df = pd.DataFrame(np.abs(np.diff(biases, axis=1)), index=amdata.var.index, columns=n_sites_label)
ax=sns.boxplot(df, color=colors[0], showfliers=False)
ax.set_ylabel('Absolute difference in bias')

# %%
# Infer accelerations and biases
ab_maps = modelling_bio.person_model(amdata, method='map', show_progress=True)
amdata.var['acc'] = ab_maps['acc']
amdata.var['bias'] = ab_maps['bias']

sns.scatterplot(data=amdata.var, x='acc', y='bias')

# %%
# Some preprocessing for the plots
person_ids = amdata.var.ID.unique().tolist()
person_id = amdata.var.ID.unique()[1]
amdata.var[amdata.var.ID==person_id].acc

# %%
# Our clock
for person in person_ids[:50]:
    sns.lineplot(x=amdata.var.WAVE, y=amdata.var[amdata.var.ID==person].acc)

# %%

# Comparison with other clocks
clocks = ['DNAmAge', 'DNAmAgeHannum','DNAmPhenoAge','DNAmGrimAge']
clock_accs = [f'{clock}Acc' for clock in clocks]
for clock in clocks:
    amdata.var[f'{clock}Acc'] = amdata.var.age - amdata.var[clock]

axs = plot.tab(clocks, ncols=2, row_size=5)
for i, clock in enumerate(clocks):
    for person in person_ids[:50]:
        sns.lineplot(x=amdata.var.WAVE, y=amdata.var[amdata.var.ID==person][f'{clock}Acc'], ax=axs[i])

# %%
# Boxplots to show mean absolute difference within the clock
clock_accs = ['acc'] + clock_accs
amdata.var[clock_accs] = (amdata.var[clock_accs]-np.min(amdata.var[clock_accs]))/(np.max(amdata.var[clock_accs])-np.min(amdata.var[clock_accs]))

all_diffs = pd.DataFrame()
for clock_acc in ['acc'] + clock_accs:
    diff = pd.DataFrame(
        # amdata.var.groupby('ID')['acc'].apply(lambda accs: pd.Series(np.abs(np.diff(accs)).T))
        amdata.var.groupby('ID')[clock_acc].apply(lambda accs: np.mean(np.abs(np.diff(accs))))
    )
    diff.rename(columns={ diff.columns[0]: 'Mean absolute difference' }, inplace = True)
    diff['clock'] = clock_acc
    all_diffs = pd.concat([all_diffs, diff])


ax=sns.boxplot(all_diffs, x='Mean absolute difference', y='clock', showfliers=False)

# %%
accs = amdata.var[amdata.var.ID==person].acc
accs = pd.DataFrame({person: amdata.var[amdata.var.ID==person].acc.values}).T
pd.DataFrame(np.abs(np.diff(accs))

accs = amdata.var.iloc[:10].acc
df = pd.DataFrame(np.abs(np.diff(accs, axis=1)), index=amdata.var.index, columns=n_sites_label)

amdata.var
for clock in clocks:
    amdata.var[f'{clock}Acc'] = amdata.var.age - amdata.var[f'DNAm{clock}']

amdata.var.groupby('ID')['acc'].agg(['min','max'])
amdata.var.groupby('ID')['acc'].var().hist()
amdata.var.groupby('ID')['AgeAcc'].var().hist()
amdata.var.groupby('ID')['AgeHannumAcc'].var().hist()
amdata.var.groupby('ID')['PhenoAgeAcc'].var().hist()
amdata.var.groupby('ID')['GrimAgeAcc'].var().hist()

















model = person_model(amdata)

with model:
    # trace =pm.sample(init='adapt_diag')
    map = pm.find_MAP()

az.plot_posterior(trace)
print(map)