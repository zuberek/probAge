# %% ########################
# %% LOADING

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from scipy.stats import linregress
from sklearn.feature_selection import r_regression

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_meta.h5ad', backed='r')
amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_meta.h5ad')

sites = pd.read_csv('../resources/AllGS_CpGs.csv')
sites = sites[sites['Clock'] == 'DeepMAge']
sites = sites.set_index('CpGs')

intersection = amdata.obs.index.intersection(sites.index)

values = np.argwhere(~np.isnan(amdata.var.units)).flatten()

amdata = amdata[intersection].to_memory()
amdata = amdata[:, values].copy()
# %%

site_name = sites.iloc[0]['CpGs']

coef = r_regression(X=amdata.X.T, y=amdata.var.units)

amdata.obs['unit_corr'] = coef**2
sns.histplot(amdata.obs.unit_corr)

amdata.obs.sort_values('unit_corr')

cg09067967