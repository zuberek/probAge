# %% ########################
# %% LOADING

# %load_ext autoreload 
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_meta.h5ad', backed='r')
wave3_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave3_participants.csv', index_col='Basename')
wave3_participants.columns
amdata = amdata['cg09067967'].to_memory()
amdata.var['heavy_drinker'] = amdata.var.units > amdata.var.units.std()
amdata.var.units = amdata.var.units/amdata.var.units.std()

amdata.var.heavy_drinker.value_counts()

# %% PLOT

PALLETE = sns.diverging_palette(220, 20, as_cmap=True)
# PALLETE = sns.diverging_palette(191, 7, as_cmap=True)

g = sns.JointGrid()
amdata = amdata[:, amdata.var.sort_values('units', ascending=True).index]
sns.scatterplot(ax=g.ax_joint, x=amdata.var.age, y=amdata['cg09067967'].X.flatten(), 
                hue=amdata.var.units, palette=PALLETE, alpha=0.7, linewidth=0, legend=False)
sns.boxplot(ax=g.ax_marg_y, y=amdata['cg09067967'].X.flatten(), x=amdata.var.heavy_drinker,
            showfliers=False)

# %% STATISTICS
from scipy.stats import f_oneway
f_oneway(amdata['cg09067967', amdata.var.heavy_drinker==True].X.flatten(),
         amdata['cg09067967', amdata.var.heavy_drinker==False].X.flatten())

from scipy.stats import ttest_ind
ttest_ind(amdata['cg09067967', amdata.var.heavy_drinker==True].X.flatten(),
         amdata['cg09067967', amdata.var.heavy_drinker==False].X.flatten())


# %%
