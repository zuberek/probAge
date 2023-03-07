import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
import arviz as az

amdata = ad.read_h5ad('../exports/wave3_acc.h5ad')

participants = amdata.var
participants['data'] = 'Every participant'

g=sns.jointplot(data=participants, x='acc', y='bias', hue='data',
                    palette=['tab:grey'], marker='.', alpha=0.3, legend=False,
                    marginal_ticks=True)
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=6, legend=False)
g.refline(y=0, x=0)

#%%
g = sns.JointGrid()
sns.scatterplot(data=amdata.var, alpha=0.5, x='acc', y='bias', color='tab:grey', ax=g.ax_joint)
sns.kdeplot(data=amdata.var, x='acc', y='bias', color='black',  ax=g.ax_joint)
sns.kdeplot(data=amdata.var, x='acc', color='black', ax=g.ax_marg_x, legend=False)
sns.kdeplot(data=amdata.var, y='bias', color='black', ax=g.ax_marg_y, legend=False)
g.refline(x=0,y=0)

g.fig.set_figwidth(6)
g.fig.set_figheight(4)

g.savefig('../results/3_jointplot_acc_bias_distribution.svg')
g.savefig('../results/3_jointplot_acc_bias_distribution.png')
#%%