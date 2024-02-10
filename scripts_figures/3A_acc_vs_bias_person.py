#%%
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling_bio
import arviz as az

import pickle

amdata = ad.read_h5ad('../exports/wave3_person_fitted.h5ad')

mean, variance = modelling_bio.bio_model_stats(amdata[0], t=np.linspace(0,100, 100))

# modelling_bio.bio_model_plot(amdata[0])

top_sites = amdata.obs.sort_values('eta_0').iloc[[0, -2]].index.values

#%%

fig, axes = plt.subplots(nrows=1, ncols=2)
legend = [False, True]


for i, site_index in enumerate(top_sites):

    non_biased = set(np.where(np.abs(amdata.var.bias)<0.005)[0])
    biased = set(np.where(amdata.var.bias>0.03)[0])
    non_acc = set(np.where(np.abs(amdata.var.acc)<0.1)[0])
    acc = set(np.where(amdata.var.acc > 0.4)[0])

    amdata.var.acc.hist()
    acc_non_biased_index = list(acc.intersection(non_biased))[3]
    biased_non_acc_index = list(biased.intersection(non_acc))[0]

    acc_non_biased_index

    bias_person = amdata[:, biased_non_acc_index].var[['acc', 'bias']]
    acc_person = amdata[:, acc_non_biased_index].var[['acc', 'bias']]
    bias_data = amdata[:, amdata.var.index == bias_person.index[0]]
    acc_data = amdata[:, amdata.var.index == acc_person.index[0]]


    # Plot
    t = np.linspace(20, 90, 100)

    mean, variance = modelling_bio.bio_model_stats(amdata[site_index], t=t)

    acc, variance = modelling_bio.bio_model_stats(amdata[site_index], t=t,
                                    acc=acc_person.acc[0],
                                    bias = acc_person.bias[0])

    bias, variance = modelling_bio.bio_model_stats(amdata[site_index], t=t,
                                    acc=bias_person.acc[0],
                                    bias = bias_person.bias[0])

    sns.scatterplot(x=amdata.var.age,
                    y=amdata[site_index].X.flatten(),
                    color='tab:grey',
                    alpha=0.1,
                    label='data',
                    legend=False,
                    ax= axes[i])
    sns.lineplot(x=t, y=mean,
                 color='tab:grey',
                 label='cohort mean',
                 legend=False,
                 ax= axes[i])

    sns.lineplot(x=t, y=acc,
                 color='tab:orange',
                 label='acc_trajectory',
                 legend=False,
                 ax=axes[i])
    sns.lineplot(x=t, y=bias,
                 color='tab:blue',
                 label='biased trajectory',
                 legend=False,
                 ax=axes[i])

    sns.scatterplot(x=[acc_data.var.age[0]],
                    y=acc_data[site_index].X.flatten(),
                    s=30, color="tab:orange",
                    ax=axes[i])

    sns.scatterplot(x=[bias_data.var.age[0]],
                    y=bias_data[site_index].X.flatten(),
                    s=30, color="tab:blue",
                    ax=axes[i])
axes[0].set_ylabel('methylation')

#%%
# # Adjusting the sub-plots
fig.set_figwidth(10)
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig('../results/Modelling/acc_vs_bias.svg')

#%%