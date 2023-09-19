
# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths
import src.preprocess_func  as preprocess_func
from sklearn.feature_selection import r_regression
import scipy.stats

# %%
# LOAD
methylation = pd.read_csv(paths.DATA_RAW)
patients = pd.read_csv(paths.DATA_PATIENTS)


methylation = methylation.set_index('ProbeID')
# %%
# CREATE ANNDATA
amdata = ad.AnnData(X= methylation.values,
        dtype=np.float32,
        obs= pd.DataFrame(index=methylation.index),
        var= patients.set_index('id'))

# %%
# PREPROCESS
amdata = preprocess_func.drop_nans(amdata)
amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)

def spearman_r_loop(site_idx):
    spr = stats.spearmanr(amdata[site_idx].X.flatten(), amdata.var.age)
    return spr.statistic

with Pool(15) as pool:
    result = list(tqdm(pool.imap(spearman_r_loop, amdata.obs.index), total=amdata.shape[0]))

stats.spearmanr(amdata.X, ages, axis=None)

# %%
# SAVE
amdata.write_h5ad(paths.DATA_PROCESSED)
# %%

def make_anndata(raw_data_path = paths.DATA_RAW, 
                 patient_data_path=paths.DATA_PATIENTS, 
                 processed_data_path = paths.DATA_PROCESSED):
        
        methylation = pd.read_csv(paths.DATA_RAW)
        patients = pd.read_csv(paths.DATA_PATIENTS)


        methylation = methylation.set_index('ProbeID')
        amdata = ad.AnnData(X= methylation.values,
                dtype=np.float32,
                obs= pd.DataFrame(index=methylation.index),
                var= patients.set_index('id'))

        amdata = preprocess_func.drop_nans(amdata)
        amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
        amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)

        amdata.write_h5ad(paths.DATA_PROCESSED)