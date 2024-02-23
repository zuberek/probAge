# Requires running
# external script for hannum dataset

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
external = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/{EXT_DSET_NAME}_person_fitted.h5ad', backed='r')
# %%
# Optional: calculate offsets for all the sites to select the most interesting one
# offset_map = bc.site_offsets(amdata, show_progress=True, map_method='Powell')
# amdata.obs['offset'] = offset_map['offset']
# amdata.obs.sort_values('offset')

# %%
# Select the interesting site (top offset in Hannum) 
SITE_NAME = 'cg00048759'

site_data = external[SITE_NAME].to_memory()

# %%
# Optional: Compute offsets if you didn't run the external script
# offset_map = bc.site_offsets(site_data, show_progress=True, map_method='Powell')
# site_data.obs['offset'] = offset_map['offset']

site_map = modelling.site_MAP(site_data, progressbar=True)
params = list(modelling.SITE_PARAMETERS.values())

xlim = (site_data.var.age.min(), site_data.var.age.max())
t = np.linspace(xlim[0],xlim[1], 1_000)

mean_after, _ = modelling.bio_model_stats(site_data, t) 

# revert the correction
site_data.obs.eta_0 = site_data.obs.eta_0 - site_data.obs.offset
site_data.obs.meth_init  = site_data.obs.meth_init - site_data.obs.offset
mean_before, _ = modelling.bio_model_stats(site_data, t) 

site_data.obs[params] = [site_map[param][0] for param in params]
mean_truth, _ = modelling.bio_model_stats(site_data, t) 

# %% ##############################
# PLOT
ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.despine()
ax.set_ylabel('Methylation level \n' + r'($\beta$-value)')
ax.set_xlabel('Age (years)')
sns.scatterplot(ax=ax, x=site_data.var.age, y=site_data.X[0].flatten(), 
                color='tab:grey', alpha=0.3 )

ax.set_ylim([0,1])


# plot truth
sns.lineplot(x=t, y=mean_truth, color='tab:blue', label='Retrained',ax=ax)

# plot model before correction
sns.lineplot(x=t, y=mean_before, color='tab:green', label='Not corrected',ax=ax)

# plot model after correction
sns.lineplot(x=t, y=mean_after, color='tab:red', label='Corrected', dashes=[5,3] ,ax=ax)




# plot model before correction

ax.get_figure().tight_layout()

# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig3/3E_model_corrected_site.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig3/3E_model_corrected_site.svg')
