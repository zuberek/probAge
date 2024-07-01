# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

from src import modelling_bio_beta as modelling
from src import batch_correction as bc
import os.path

N_SITES = None
N_PARTICIPANTS = None

EXT_DSET_NAME = 'downsyndrome' # external dataset name
REF_DSET_NAME = 'wave3' # reference datset name


hannum_external = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/hannum_person_fitted.h5ad', backed='r')
wave3 = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/wave3_person_fitted.h5ad', backed='r')

# %%

site_index = hannum_external.obs.offset.abs().sort_values().index[-1]

plot.fonts(8)
ax = plot.row(site_index, figsize=(9.5, 6))
sns.despine()

# show the offset applied to data

sns.scatterplot(s=15, color=colors[2], x=wave3.var.age, y=wave3[site_index].X.flatten(), label='Generation\nScotland')
sns.scatterplot(s=15, color=colors[5], x=hannum_external.var.age, y=hannum_external[site_index].X.flatten(), label='Hannum')
sns.scatterplot(s=15, color=colors[1], x=hannum_external.var.age, y=hannum_external[site_index].X.flatten()-hannum_external[site_index].obs.offset.values, label='Hannum\ncorrected')

ax.set_xlabel("Age (years)")
ax.set_ylabel('Methylation level \n' + r'($\beta$-value)')

ax.legend(loc='best', markerscale=1.5)

ax.get_figure().tight_layout()

# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext4/Ext4A_corrected_data.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext4/Ext4A_corrected_data.svg')

