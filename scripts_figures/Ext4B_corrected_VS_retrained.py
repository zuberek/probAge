# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

from src import modelling_bio_beta as model
from src import batch_correction as bc
import os.path

hannum_external = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/hannum_person_fitted.h5ad', backed='r')
plot_df = pd.read_csv('../exports/hannum_retraining.csv', index_col=0)
# %%
# prep
plot_df = pd.DataFrame()

df = pd.DataFrame()
# df[['Corrected Difference Acc', 'Corrected Difference Bias']] = plot_df[plot_df.model =='Corrected'][['acc', 'bias']] - plot_df[plot_df.model =='Retrained'][['acc', 'bias']]
df[['acc', 'bias']] = hannum_external.var[['acc_wave3' , 'bias_wave3']].values - hannum_external.var[['retrained_acc' , 'retrained_bias']].values
df['model']='Corrected'
plot_df = pd.concat([plot_df, df], axis='index')  

df = pd.DataFrame()
df[['acc', 'bias']] = hannum_external.var[['not_corrected_acc' , 'not_corrected_bias']].values - hannum_external.var[['retrained_acc' , 'retrained_bias']].values
df['model']='Not Corrected'
plot_df = pd.concat([plot_df, df], axis='index')  
plot_df

# %%
# plot
g = sns.JointGrid()
plot.fonts(8)


sns.scatterplot(ax=g.ax_joint, data=plot_df, x='acc',  y= 'bias', hue='model',
                alpha=0.6, palette=colors[0:2], s=20)
sns.kdeplot(linewidth=1, ax=g.ax_marg_y, data=plot_df,  y= 'bias', hue='model',
                 palette=colors[0:2], legend=False)
sns.kdeplot(linewidth=1, ax=g.ax_marg_x, data=plot_df,  x= 'acc', hue='model',
                 palette=colors[0:2], legend=False)


g.refline(x=0, y=0)
g.ax_joint.set_ylabel(r'$\Delta$-bias ($\beta$-value)')
g.ax_joint.set_xlabel(r'$\Delta$-acceleration ($\beta$-value/year)')

g.figure.set_size_inches((plot.cm2inch(9.5),plot.cm2inch(6)))
g.figure.tight_layout()

legend = g.ax_joint.legend(loc='center right')
for handle in legend.legendHandles:
    handle.set_alpha(1)

# %% saving
g.savefig(f'{paths.FIGURES_DIR}/ext4/Ext4B_corrected_VS_retrained.svg')
g.savefig(f'{paths.FIGURES_DIR}/ext4/Ext4B_corrected_VS_retrained.png')
