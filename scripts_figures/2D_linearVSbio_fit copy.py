# %% ########################
# LOADING

%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import src.modelling_bio_beta as modelling
from scipy import stats

import plotly.express as px
import numpy as np
import pandas as pd
import pickle

amdata = ad.read_h5ad('../exports/wave3_person_fitted.h5ad', backed='r')
amdata = ad.read_h5ad('../exports/wave3_sites_fitted.h5ad', backed='r')

amdata.obs.sort_values('spr2')

# %%
top_sites = amdata.obs.sort_values('omega').index[-12:]
axes = plot.tab(top_sites, row_size=10)
for i, ax in enumerate(axes):
    plot_comparison(top_sites[i], ax=ax)

# plot_comparison(SITE_NAME, ax=axes[0])

# %%
SITE_NAME = 'cg13108341' 

def plot_comparison(SITE_NAME, ax):
    site_data = amdata[SITE_NAME].to_memory()

    linfit = stats.linregress(x=site_data.var.age, y=site_data.X.flatten())

    # ax = plot.row(f'{SITE_NAME}\nELOVL2', figsize=(9.5, 7))
    plot.fonts(8)
    sns.despine()

    sns.scatterplot(x=site_data.var.age, y=site_data.X.flatten(),
                    color='tab:grey', label="Data", alpha=0.2, s=10, ax=ax)

    legend = ax.legend(loc='best')
    for handle in legend.legendHandles:
        handle.set_alpha(1)
        # handle.set_markersize(1)

    xlim = (0, 120)
    ax.set_xlim([0,120])

    # ax.set_ylim([0.2,1])
    ax.set_ylim([0,1])
    ax.set_ylabel('Methylation level \n' +r'($\beta$-value)')
    ax.set_xlabel('Age (years)')

    t = np.linspace(xlim[0],xlim[1], 1_000)
    # Add linear fit
    mean_y = linfit.slope*np.array([xlim[0],xlim[1]]) + linfit.intercept
    std2_plus = mean_y+2*np.sqrt(0.001)
    std2_minus = mean_y-2*np.sqrt(0.001)
    sns.lineplot(x=[xlim[0],xlim[1]], y=mean_y, linewidth=1, ax=ax, label='Linear', color=colors[0])
    sns.lineplot(x=[xlim[0],xlim[1]], dashes=[5,2], y=std2_plus, linewidth=1, ax=ax, color=colors[0])
    sns.lineplot(x=[xlim[0],xlim[1]], dashes=[5,2], y=std2_minus, linewidth=1, ax=ax, color=colors[0])

    # Add bio fit
    mean, variance = modelling.bio_model_stats(site_data, t)

    k = (mean*(1-mean)/variance)-1
    a = mean*k
    b = (1-mean)*k

    conf_int = np.array(stats.beta.interval(0.95, a, b))
    low_conf = conf_int[0]
    upper_conf = conf_int[1]    
        
    sns.lineplot(x=t, y=mean, color=colors[1], label='Bio',linewidth=1, ax=ax)
    sns.lineplot(x=t, y=low_conf, dashes=[5,2], color=colors[1], label='2-std',linewidth=1, ax=ax)
    sns.lineplot(x=t, y=upper_conf, dashes=[5,2], color=colors[1],linewidth=1, ax=ax)

    import matplotlib.lines as mlines

    bio_handle = mlines.Line2D([], [], color=colors[0], marker='o', linestyle='None',
                            markersize=5, label='Bio')
    lin_handle = mlines.Line2D([], [], color=colors[1], marker='o', linestyle='None',
                            markersize=5, label='Linear')
    mean_handle = mlines.Line2D([], [], color='tab:grey', linestyle='-',
                            markersize=5, label='Mean')
    std_handle = mlines.Line2D([], [], color='tab:grey', linestyle='--',
                            markersize=5, label='95% CI')

    plt.legend(handles=[bio_handle, lin_handle, mean_handle, std_handle])


    ax.get_figure().tight_layout()

# legend.legendHandles[0].set_sizes(5)

# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig2/2D_linearVSbio_fit.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig2/2D_linearVSbio_fit.svg', transparent=False)

# %%
