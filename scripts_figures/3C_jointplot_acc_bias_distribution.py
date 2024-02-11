
#%%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling_bio
import arviz as az

amdata = ad.read_h5ad('../exports/wave3_person_fitted.h5ad', backed='r')

participants = amdata.var
participants['data'] = 'Every participant'

#%%
g = sns.JointGrid()
sns.scatterplot(data=amdata.var, alpha=0.5, x='acc', y='bias', color='tab:grey', ax=g.ax_joint)
sns.kdeplot(data=amdata.var, x='acc', y='bias', color='black',  ax=g.ax_joint)
sns.kdeplot(data=amdata.var, x='acc', color='black', ax=g.ax_marg_x, legend=False)
sns.kdeplot(data=amdata.var, y='bias', color='black', ax=g.ax_marg_y, legend=False)
g.refline(x=0,y=0)

g.fig.set_figwidth(6)
g.fig.set_figheight(4)

# %% saving
g.savefig(f'{paths.FIGURES_DIR}/3C_jointplot_accbias_distribution.svg')
g.savefig(f'{paths.FIGURES_DIR}/3C_jointplot_accbias_distribution.png')
#%%