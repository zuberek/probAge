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

# %%
SITE_NAME = 'cg16867657' 
site_data = amdata[SITE_NAME].to_memory()

linfit = stats.linregress(x=site_data.var.age, y=site_data.X.flatten())

# %%
ax = plot.row(f'{SITE_NAME}\nELOVL2', figsize=(9.5, 6))
plot.fonts(8)
sns.despine()

sns.scatterplot(x=site_data.var.age, y=site_data.X.flatten(),
                color='tab:grey', alpha=0.2, size=1)

ax.set_ylim([0,1])
ax.set_ylabel('Methylation level \n(beta value)')
ax.set_xlabel('Age')
xlim = (amdata.var.age.min(), amdata.var.age.max())

# Add bio fit
t = np.linspace(xlim[0],xlim[1], 1_000)

mean, variance = modelling.bio_model_stats(site_data, t)

k = (mean*(1-mean)/variance)-1
a = mean*k
b = (1-mean)*k

conf_int = np.array(stats.beta.interval(0.95, a, b))
low_conf = conf_int[0]
upper_conf = conf_int[1]    
    
sns.lineplot(x=t, y=mean, color='tab:blue', label='mean',ax=ax)
sns.lineplot(x=t, y=low_conf, color='tab:cyan', label='2-std',ax=ax)
sns.lineplot(x=t, y=upper_conf, color='tab:cyan',ax=ax)


# Add linear fit
mean_y = linfit.slope*np.array([xlim[0],xlim[1]]) + linfit.intercept
std2_plus = mean_y+2*np.sqrt(0.001)
std2_minus = mean_y-2*np.sqrt(0.001)
sns.lineplot(x=[xlim[0],xlim[1]], y=mean_y, ax=ax, label='linear_mean', color='tab:orange')
sns.lineplot(x=[xlim[0],xlim[1]], y=std2_plus, ax=ax, color='tab:orange', label='linear_2-std')
sns.lineplot(x=[xlim[0],xlim[1]], y=std2_minus, ax=ax, color='tab:orange', label='linear_2-std')

ax.get_figure().tight_layout()
plt.legend(title='Models')

# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig2/2D_linearVSbio_fit.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig2/2D_linearVSbio_fit.svg')
