# %%

%load_ext autoreload
%autoreload 2
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
import pymc as pm

import arviz as az
from operator import itemgetter

import pickle

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

# %%
# Load bio sites
amdata_hannum = amdata_src.AnnMethylData('../exports/hannum_acc_bio.h5ad')

keep_sites = list(np.arange(0, amdata_hannum.shape[0]))
[keep_sites.remove(i) for i in [8, 172]]
amdata_hannum = amdata_hannum[keep_sites]

modelling_bio.bio_model_plot(amdata_hannum[8])


res = modelling_bio.person_model(amdata_hannum[:10, :],
                        show_progress=True,
                        return_MAP=True,
                        return_trace=False)

res['map']['acc'] = np.log2(res['map']['acc'])
amdata_hannum.var['acc_mean'] = res['map']['acc']
amdata_hannum.var['bias_mean'] = res['map']['bias']
sns.jointplot(amdata_hannum.var, x='acc_mean', y='bias_mean')

amdata_hannum.write_h5ad('../exports/hannum_acc_bio.h5ad')

# %%
# Load site fits
fits_bio = pd.read_csv('../exports/fits_bio.csv')
comparisons = pd.read_csv('../exports/comparison_bio.csv', index_col=[0,1])
# Load bio sites
amdata = amdata_src.AnnMethylData('../exports/wave3_bio.h5ad')

bio_model = comparisons.xs('bio', level='model')

sns.histplot(bio_model, x='elpd_loo')


site = bio_model.index[np.argmin(bio_model.elpd_loo)]

for i in range(20):
    modelling_bio.bio_model_plot(amdata[sites[i]])
    plt.show()