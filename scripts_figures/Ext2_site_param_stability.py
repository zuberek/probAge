
# %% ########################
### LOADING

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import modelling_bio_beta as modelling
import pymc as pm
import pickle

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_meta.h5ad', backed='r')

SITE_NAME = 'cg16867657'
amdata = amdata[SITE_NAME].to_memory()

# %% ########################
### MODELLING

model = modelling.bio_sites(amdata[SITE_NAME])
with model:
    trace = pm.sample(1000, tune=1000, progressbar=True)
with open(f'{paths.DATA_PROCESSED_DIR}/wave3_{SITE_NAME}_trace.pickle',"wb") as file:
    pickle.dump(trace,file)
    file.close()

# %%
with open(f'{paths.DATA_PROCESSED_DIR}/wave3_{SITE_NAME}_trace.pickle',"rb") as file:
    trace = pickle.load(file)
    file.close() 


# %%
params = list(modelling.SITE_PARAMETERS.values())
axes = plot.tab(subtitles=params, ncols=2)
for i, param in enumerate(params):
    sns.kdeplot(x=az.extract(trace.posterior)[param].values[0], ax=axes[i])
# %%
summary = az.summary(trace)
az.summary(trace, round_to=5)
summary = summary.set_index(pd.Series(summary.index.values.astype('str')).str.split('[', expand=True)[0])

df = summary.iloc[0][['mean', 'sd']]
sns.boxplot(data=[94603, 113524, 132444, 151365, 170285])

axes = plot.tab(subtitles=params, ncols=2)
for i, param in enumerate(params):
    sns.kdeplot(x=az.extract(trace.posterior)[param].values[0], ax=axes[i])

az.plot_trace(trace)
az.plot_posterior(trace)
amdata_site = amdata[SITE_NAME].copy()
amdata_site.obs[params] = summary.loc[params, 'mean'].values
modelling.bio_model_plot(amdata_site)

# %%
maps= participants.loc[person_names][['acc', 'bias']]

g = sns.JointGrid(marginal_ticks=True,)
sns.scatterplot(data=extracted_df, y='bias', x='acc', ax=g.ax_joint,
                marker='.', alpha=0.3, hue='person', legend=False)
sns.scatterplot(data=participants.loc[person_names].reset_index(), x='acc', y='bias', 
                hue='index', legend=False, ax=g.ax_joint)                
sns.kdeplot(data=extracted_df, y='bias', x='acc', hue='person', 
                ax=g.ax_joint, thresh=0.25, legend=False, fill=False)
sns.kdeplot(data=extracted_df,  x='acc', common_norm=True, legend=False, hue='person',ax=g.ax_marg_x)
sns.kdeplot(data= extracted_df, y='bias', common_norm=True, legend=False, hue='person',ax=g.ax_marg_y)
g.ax_marg_x.vlines(maps.values[:,0], ymin=0, ymax=1, colors=colors)
g.ax_marg_y.hlines(maps.values[:,1], xmin=0, xmax=20, colors=colors)

g.fig.set_figwidth(6)
g.fig.set_figheight(4)