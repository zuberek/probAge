%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *


wave3_drift = pd.read_csv('../exports/wave3_participants.csv')

wave3_bio = ad.read_h5ad('../exports/wave3_acc_bio.h5ad')
hannum = ad.read_h5ad('../exports/hannum.h5ad')

ax=sns.scatterplot(x=wave3_bio.var.acc_mean.values, y=wave3_drift.acc_mean.values)
sns.lineplot(x=[-1,1], y=[-1,1], ax=ax)
ax.set_ylabel('Wave3 drift acc')
ax.set_xlabel('Wave3 bio acc')

ax=sns.scatterplot(x=wave3_bio.var.bias_mean.values, y=wave3_drift.bias_mean.values)
sns.lineplot(x=[-0.1,0.1], y=[-0.1,0.1], ax=ax)
ax.set_ylabel('Wave3 drift bias')
ax.set_xlabel('Wave3 bio bias')
ax=sns.scatterplot(x=wave3_bio.var.bias_mean.values, y=wave3_drift.bias_mean.values)
sns.lineplot(x=[-0.1,0.1], y=[-0.1,0.1], ax=ax)
ax.set_ylabel('Wave3 drift bias')
ax.set_xlabel('Wave3 bio bias')

wave_bio_fits = pd.read_csv('../exports/fits_bio.csv')
sns.histplot(wave_bio_fits.sd, log_scale=True, bins=20)

hannum.var.age.hist() 