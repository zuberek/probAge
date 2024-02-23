
# %% ########################
### LOADING

# %load_ext autoreload 
# %autoreload 2

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
params = list(modelling.SITE_PARAMETERS.values())

# %%
params_latex=['$\eta_0$', '$\omega$', '$p$', '$S$', '$c$']

# %%
params = list(modelling.SITE_PARAMETERS.values())
plot.fonts(8)
axes = plot.tab(subtitles=params_latex, ncols=3, figsize=[19,12.5])
summary   = az.summary(trace)
summary = summary.set_index(pd.Series(summary.index.values.astype('str')).str.split('[', expand=True)[0])
# left, right = summary.loc[param][['hdi_3%', 'hdi_97%']]
# sns.lineplot(x=[])
for i, param in enumerate(params):
    sns.kdeplot(x=az.extract(trace.posterior)[param].values[0], ax=axes[i])
    sns.despine()
axes[-1].remove()
plt.tight_layout()
# %%
# Save
axes[0].get_figure().savefig(f'{paths.FIGURES_DIR}/ext2/Ext2A_site_param_stability.png')
axes[0].get_figure().savefig(f'{paths.FIGURES_DIR}/ext2/Ext2A_site_param_stability.svg')









# %%
figure = plt.figure()
axes = az.plot_posterior(trace, var_names=params)
figure.add_axes(axes[0][0])
axes[0][0]
figsize = (plot.cm2inch(9.5), 
           plot.cm2inch(6))
fig, ax = plt.subplots(figsize=figsize)
ax.plot(range(10))

fig2 = plt.figure()
fig2.axes.append(ax)

plt.show()


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