# %% 
# IMPORTS

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import src.modelling_bio_beta as modelling
import src.batch_correction as bc

EXT_DSET_NAME = 'hannum' # external dataset name
REF_DSET_NAME = 'wave3' # reference datset name


# %%
# LOADING

# amdata = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{EXT_DSET_NAME}_fitted.h5ad', backed='r')
hannum = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{EXT_DSET_NAME}_meta.h5ad', backed='r')
amdata_ref = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{REF_DSET_NAME}_person_fitted.h5ad', backed='r')
hannum_external = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{EXT_DSET_NAME}_person_fitted.h5ad', backed='r')

hannum_participants = hannum_external.var

hannum = bc.merge_external(hannum, amdata_ref)

# %%
# PREP

# revert the correction and calculate the wrong acc and bias
hannum_external.obs.eta_0 = hannum_external.obs.eta_0 - hannum_external.obs.offset
hannum_external.obs.meth_init  = hannum_external.obs.meth_init - hannum_external.obs.offset
maps = modelling.person_model(hannum_external[0], progressbar=True)
hannum_external.var['wrong_acc'] = maps['acc']
hannum_external.var['wrong_bias'] = maps['bias']

# retrain a new model from scratch on hannum
params = list(modelling.SITE_PARAMETERS.values())
site_maps = []
for site in tqdm(hannum.obs.index):
    site_maps.append(modelling.site_MAP(hannum[site], progressbar=False))

for param in  params:
    hannum.obs[param] = site_maps[param]

hannum.obs[params] = [site_maps[param] for param in params]
person_maps = modelling.person_model(hannum)

# np.sum(hannum.X>1)


# %%
# PLOTTING
# plt.figure(figsize=(plot.cm2inch(9.5),pl2ot.cm2inch(6)))
g = sns.JointGrid()


sns.scatterplot(ax=g.ax_joint, x=hannum_external.var.acc_wave3,  y= hannum_external.var.bias_wave3,
                alpha=0.1, color='tab:red', label='Corrected')
sns.kdeplot(ax=g.ax_joint, x=hannum_external.var.acc_wave3,  y= hannum_external.var.bias_wave3,
                levels=5, color='tab:red', label='Corrected')

sns.scatterplot(ax=g.ax_joint, x=hannum_external.var.wrong_acc,  y= hannum_external.var.wrong_bias,
                alpha=0.1, color='tab:green', label='Not Corrected')
sns.kdeplot(ax=g.ax_joint, x=hannum_external.var.wrong_acc,  y= hannum_external.var.wrong_bias,
                levels=5, color='tab:green', label='Not Corrected')

sns.kdeplot(ax=g.ax_marg_x, x=hannum_external.var.bias_wave3,
            color='tab:red', legend=True)

g.ax_joint.set_ylabel('Bias (beta)')
g.ax_joint.set_xlabel('Acceleration (beta/year)')

plot.fonts(8)
g.figure.subplots_adjust(top=0.7)
g.figure.set_size_inches((plot.cm2inch(9.5),plot.cm2inch(6)))

g.figure.tight_layout()

# %%
g.figure.savefig(f'{paths.FIGURES_DIR}/fig3/3F_prediction_corrected.png')
g.figure.savefig(f'{paths.FIGURES_DIR}/fig3/3F_prediction_corrected.svg')
