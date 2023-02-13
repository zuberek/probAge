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