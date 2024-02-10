# %%
# IMPORTS
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling
import pickle
import arviz as az
import pymc as pm
cpgs = pd.read_csv('../resources/final_CpGs.csv')

amdata = ad.read_h5ad('../exports/wave3_person_fitted.h5ad')

amdata = amdata[amdata.obs.sort_values('spr2').index]
site_names = amdata.obs.index.values
# %%
# Compute traces


traces = []

for site_name in site_names[7:50]:
    model = modelling.bio_sites(amdata[site_name])
    with model:
        trace = pm.sample(1000, tune=1000, progressbar=True)
        pm.compute_log_likelihood(trace)

    traces.append(trace)

    filehandler = open(f'{paths.DATA_PROCESSED_DIR}/traces/{site_name}.idata',"wb")
    pickle.dump(trace,filehandler)
    filehandler.close()


# %%
# compute rhats
from os import listdir
sites = listdir('../exports/traces/')

params = list(modelling.SITE_PARAMETERS.values())
r_hats = pd.DataFrame()
for name in tqdm(sites):
    with open(f'../exports/traces/{name}', 'rb') as f:
        idata = pickle.load(f)

    summary = az.summary(idata, round_to=5)
    summary = summary.set_index(pd.Series(summary.index.values.astype('str')).str.split('[', expand=True)[0])

    r_hat = pd.DataFrame(summary.loc[params].r_hat)
    r_hat['site'] = name.split('.')[0]
    r_hat = r_hat.reset_index().rename(columns={0:'param'})
    r_hats = pd.concat((r_hats, r_hat))
    
# %%
# prep df
df = r_hats.copy()
df.loc[df.param=='eta_0', 'param'] = 'eta'
df.loc[df.param=='meth_init', 'param'] = 'p'
df.loc[df.param=='system_size', 'param'] = 'S'
df.loc[df.param=='var_init', 'param'] = 'c'
# %%
# plot
ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.despine()
sns.boxplot(ax=ax, data=r_hats, x='param', y='r_hat', showfliers=False)

ax.set_xlabel('Site parameter')
ax.set_ylabel('R_hat distribution')

plt.tight_layout()


# %%
# save

