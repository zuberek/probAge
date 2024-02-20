# %%
# IMPORTS
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from statsmodels.nonparametric.smoothers_lowess import lowess

from src import modelling_bio_beta as modelling
import pickle
import arviz as az
import pymc as pm

amdata = ad.read_h5ad('../exports/wave3_person_fitted.h5ad', backed='r')

SITE_NAME = 'cg16867657'
site_data = amdata[SITE_NAME].to_memory()

# %%
with open(f'{paths.DATA_PROCESSED_DIR}/wave3_{SITE_NAME}_trace.pickle',"rb") as file:
    trace = pickle.load(file)
    file.close() 

# %%
model = modelling.bio_sites(site_data)
pm.sample_posterior_predictive(trace, model=model, extend_inferencedata=True, )

# %%

ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.despine()
ax.set_xlabel('Age (years)')
ax.set_ylabel('Methylation value\n(beta level)')


# Plot the posterior predictive samples for full model
y_vals=trace.posterior_predictive['m-values'].sel(sites=SITE_NAME).mean(axis=0).mean(axis=0)
upper = np.percentile(trace.posterior_predictive['m-values'].sel(sites=SITE_NAME).mean(axis=0),97.5, axis=0)
lower = np.percentile(trace.posterior_predictive['m-values'].sel(sites=SITE_NAME).mean(axis=0),2.5, axis=0)


ys_mean = lowess(y_vals, site_data.var.age, return_sorted=True, frac=0.4)
ys_upper = lowess(upper, site_data.var.age,  return_sorted=True, frac=0.4)
ys_lower = lowess(lower, site_data.var.age, return_sorted=True, frac=0.4)
plt.scatter(site_data.var.age, site_data.X.T, color='tab:grey', alpha=0.05)
# Plot the 95% predictive interval
plt.plot(ys_mean[:,0], ys_mean[:,1], label='mean', color=colors[1], alpha=0.5)
plt.plot(ys_upper[:,0], ys_upper[:,1], label='std', color=colors[1], dashes=[5,3], alpha=0.5)
plt.plot(ys_upper[:,0], ys_lower[:,1], color=colors[1], dashes=[5,3], alpha=0.5)
plt.legend()
plt.tight_layout()
# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext2/Ext2C_predictive_posterior.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext2/Ext2C_predictive_posterior.svg')
