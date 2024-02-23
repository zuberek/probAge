# %% 
# IMPORTS

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import src.modelling_bio_beta as modelling

DSET_NAME = 'downsyndrome' # reference datset name

amdata = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{DSET_NAME}_person_fitted.h5ad', backed='r')
amdata.var.status = amdata.var.status.astype('string')
amdata.var.loc[amdata.var.status=='healthy', 'status'] = 'Control'
# %% 
# PLOT
plot.fonts(8)
g = sns.JointGrid()
sns.scatterplot(ax=g.ax_joint, data=amdata.var, x='bias_wave3', y='acc_wave3', hue='status',
                 alpha=1, linewidth=0, s=20, legend=True)
g.ax_joint.legend(title=None)

sns.boxplot(data=amdata.var, x='bias_wave3', y='status',palette=[colors[0], colors[1]], ax=g.ax_marg_x, showfliers=False, legend=False)
sns.boxplot(data=amdata.var, y='acc_wave3', x='status',palette=[colors[0], colors[1]], ax=g.ax_marg_y, showfliers=False, legend=False)

g.ax_joint.set_ylabel('Acceleration (beta/year)')
g.ax_joint.set_xlabel('Bias (beta)')


x1, x2 = 0, 1 
y, h, col = amdata.var['acc_wave3'].max() + 0.03, 0.02, 'k'
g.ax_marg_y.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
g.ax_marg_y.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)


g.figure.set_size_inches((plot.cm2inch(9.5),plot.cm2inch(6)))
g.figure.tight_layout()

# %% saving
g.savefig(f'{paths.FIGURES_DIR}/fig3/3H_downsyndrome_results.svg')
g.savefig(f'{paths.FIGURES_DIR}/fig3/3H_downsyndrome_results.png')



# %% stats
from scipy.stats import ttest_ind

ttest_ind(amdata.var[amdata.var.status=='healthy'].acc_wave3,
         amdata.var[amdata.var.status=='Down syndrome'].acc_wave3)