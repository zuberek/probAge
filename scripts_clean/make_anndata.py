
# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
import src.preprocess_func  as preprocess_func


EXTERNAL_OPEN_PATH = '../data/Patients_betas_reduced_ready-40.csv'
EXTERNAL_PATIENTS = '../data/Annotation_paediatric_40.csv'
EXTERNAL_SAVE_PATH = '../exports/Nelly.h5ad'


# %%
# LOAD
methylation = pd.read_csv(EXTERNAL_OPEN_PATH)
patients = pd.read_csv(EXTERNAL_PATIENTS)


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

# %%
# SAVE
amdata.write_h5ad(EXTERNAL_SAVE_PATH)
# %%
