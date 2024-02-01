import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
import arviz as az

import pickle

amdata = amdata_src.AnnMethylData('../exports/wave3_acc_bio.h5ad')

# Load site fits
fits_bio = pd.read_csv('../exports/fits_bio.csv')
comparisons = pd.read_csv('../exports/comparison_bio.csv', index_col=[0,1])
# Load bio sites
amdata = amdata_src.AnnMethylData('../exports/wave3_bio.h5ad')

bio_model = comparisons.xs('bio', level='model')

sites = bio_model.sort_values(by='elpd_loo').index.values

for i in range(20):
    modelling_bio.bio_model_plot(amdata[sites[i]])
    plt.show()
