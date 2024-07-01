# %% 
# IMPORTS

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import src.modelling_bio_beta as modelling
import src.batch_correction as bc
import pymc as pm

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
# PREP1 Not corrected

# revert the correction and calculate the wrong acc and bias
maps = modelling.person_model(hannum, progressbar=True)
maps['bias']

def single_person_momdel(index):
    return modelling.person_model(hannum[:, index])

with Pool(7) as p:
    maps =list(tqdm(p.imap(single_person_momdel,
                                      hannum.var.index
                                     )
                                ,total=hannum.shape[1]
                                )
                            )
acc = [m['acc'][0] for m in maps]
bias = [m['bias'][0] for m in maps]

hannum_external.var['not_corrected_acc'] = acc #maps['acc']
hannum_external.var['not_corrected_bias'] = bias #maps['bias']
hannum_external.write_h5ad(f'{paths.DATA_PROCESSED_DIR}/{EXT_DSET_NAME}_person_fitted.h5ad')

# %%
# PREP2 Retrained
# retrain a new model from scratch on hannum
params = list(modelling.SITE_PARAMETERS.values())
hannum = modelling.site_MAP(hannum[0], progressbar=True)

# def singe_site(index):
#     return modelling.site_MAP(hannum[index], progressbar=False)

# with Pool(7) as p:
#     maps =list(tqdm(p.imap(singe_site,
#                                       hannum.obs.index
#                                      )
#                                 ,total=hannum.shape[0]
#                                 ))

with modelling.bio_sites(hannum):
    sites_map = pm.find_MAP(maxeval=10000, method='L-BFGS-B', progressbar=True)

params = list(modelling.SITE_PARAMETERS.values())
for param in  params:
    hannum.obs[param] = sites_map[param]


# hannum.obs[params] = [sites_map[param] for param in params]
with Pool(7) as p:
    person_maps =list(tqdm(p.imap(single_person_momdel,
                                      hannum.var.index
                                     )
                                ,total=hannum.shape[1]
                                ))

acc = [m['acc'][0] for m in person_maps]
bias = [m['bias'][0] for m in person_maps]

hannum_external.var['acc'] = acc
hannum_external.var['bias'] = bias

# np.sum(hannum.X>1)


# %%
# PREP DATA

plot_colors = {
    'corrected': 'tab:red',
    'not_corrected_acc': 'tab:green',
    'retrained': 'tab:blue',
}

plot_df = pd.DataFrame()


df = hannum_external.var[['retrained_acc', 'retrained_bias']]
df = df.rename(columns={'retrained_acc':'acc', 'retrained_bias': 'bias'})
df['model'] = 'Retrained'
plot_df = pd.concat([plot_df, df], axis='index')  

df = hannum_external.var[['not_corrected_acc', 'not_corrected_bias']]
df = df.rename(columns={'not_corrected_acc':'acc', 'not_corrected_bias': 'bias'})
df['model'] = 'Not Corrected'
plot_df = pd.concat([plot_df, df], axis='index')  


df = hannum_external.var[['acc_wave3', 'bias_wave3']]
df = df.rename(columns={'acc_wave3':'acc', 'bias_wave3': 'bias'})
df['model'] = 'Corrected'
plot_df = pd.concat([plot_df, df], axis='index')  



# %%
g = sns.JointGrid()
plot.fonts(8)


sns.scatterplot(ax=g.ax_joint, data=plot_df, x='acc',  y= 'bias', hue='model',
                alpha=0.7, palette=[colors[5], colors[2], colors[1]], s=20)
sns.kdeplot(linewidths=1, ax=g.ax_joint, data=plot_df, x='acc',  y= 'bias', hue='model',
                levels=5, palette=[colors[5], colors[2], colors[1]])
sns.kdeplot(linewidth=1, ax=g.ax_marg_y, data=plot_df,  y= 'bias', hue='model',
                 palette=[colors[5], colors[2], colors[1]], legend=False)
sns.kdeplot(linewidth=1, ax=g.ax_marg_x, data=plot_df,  x= 'acc', hue='model',
                 palette=[colors[5], colors[2], colors[1]], legend=False)

g.ax_joint.set_ylabel(r'Bias ($\beta$-value)')
g.ax_joint.set_xlabel(r'Acceleration ($\beta$-value/year)')

g.figure.set_size_inches((plot.cm2inch(9.5),plot.cm2inch(6)))
g.figure.tight_layout()

# g.ax_joint.legend(bbox_to_anchor=(0, 1), loc='upper left')
legend = g.ax_joint.legend(loc='upper left')
# dir(legend.legendHandles[0])
# dir(legend.legendHandles[0]._marker)

for handle in legend.legendHandles:
    handle.set_alpha(1)


# %% saving
g.savefig(f'{paths.FIGURES_DIR}/fig3/3F_prediction_corrected.svg', transparent=True)
g.savefig(f'{paths.FIGURES_DIR}/fig3/3F_prediction_corrected.png', transparent=True)

# %%


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

N = 10

x = np.random.rand(N)
y = np.random.rand(N)

line, = plt.plot(x, y, marker='*', markersize=20, markeredgecolor='black',
   alpha=0.4, ls='none', label='Random Data')

legend = plt.legend(loc='upper right')
legend.legendHandles[0]._legmarker.set_markersize(15)
legend.legendHandles[0]._legmarker.set_alpha(1)
