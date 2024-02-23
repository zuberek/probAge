
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
plot.fonts(8)
g = sns.JointGrid()

sns.scatterplot(data=amdata.var, alpha=0.3, x='acc', y='bias', 
                color='tab:grey', ax=g.ax_joint)
sns.kdeplot(data=amdata.var, x='acc', y='bias',  linewidths=1,
            color='black',  ax=g.ax_joint)
sns.kdeplot(data=amdata.var, x='acc', color='black', linewidth=1,
            ax=g.ax_marg_x, legend=False)
sns.kdeplot(data=amdata.var, y='bias', color='black',  linewidth=1,
            ax=g.ax_marg_y, legend=False)
g.refline(x=0,y=0)

g.ax_joint.set_ylabel(r'Bias ($\beta$-value)')
g.ax_joint.set_xlabel('Acceleration \n' + r'($\beta$-value/year)')
g.figure.set_size_inches((plot.cm2inch(9.5),plot.cm2inch(6)))

g.figure.tight_layout()

# %% saving
g.savefig(f'{paths.FIGURES_DIR}/fig3/3C_jointplot_accbias_distribution.svg')
g.savefig(f'{paths.FIGURES_DIR}/fig3/3C_jointplot_accbias_distribution.png')
#%%