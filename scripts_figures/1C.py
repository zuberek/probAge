# %% ########################
# %% LOADING

# %load_ext autoreload 
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_meta.h5ad', backed='r')
wave3_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave3_participants.csv', index_col='Basename')

amdata = amdata['cg24090911'].to_memory()
amdata.var['heavy_smoker'] = amdata.var.weighted_smoke > amdata.var.weighted_smoke.std()


# %% PLOT
g = sns.JointGrid()
PALLETE = sns.diverging_palette(220, 20, as_cmap=True)
PALLETE = sns.diverging_palette(220, 20, as_cmap=True)
# PALLETE = sns.color_palette('vlag', as_cmap=True)

# amdata = amdata[:, np.random.shuffle(amdata.var.index)].copy()
amdata = amdata[:, amdata.var.sort_values('weighted_smoke', ascending=True).index]
sns.scatterplot(ax=g.ax_joint, x=amdata.var.age, y=amdata['cg24090911'].X.flatten(), 
                hue=amdata.var.weighted_smoke, palette='crest', alpha=0.7, linewidth=0, legend=False)
sns.boxplot(ax=g.ax_marg_y, y=amdata['cg24090911'].X.flatten(), x=amdata.var.heavy_smoker,
            showfliers=False)
amdata['cg24090911'].X.flatten()

# %% STATISTICS
from scipy.stats import f_oneway
f_oneway(amdata['cg24090911', amdata.var.heavy_smoker==True].X.flatten(),
         amdata['cg24090911', amdata.var.heavy_smoker==False].X.flatten())

from scipy.stats import ttest_ind
ttest_ind(amdata['cg24090911', amdata.var.heavy_smoker==True].X.flatten(),
         amdata['cg24090911', amdata.var.heavy_smoker==False].X.flatten())


# %%
