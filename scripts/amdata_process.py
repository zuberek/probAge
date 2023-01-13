%load_ext autoreload
%autoreload 2
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from sklearn.feature_selection import r_regression

INPUT_PATH = '../exports/wave3.h5ad'
META_DIR_PATH = '/afs/inf.ed.ac.uk/user/s17/s1768506/disk/methylation/genscot_meta/wave3'

print('Loading the dataset (it might take a while)...')
amdata = ad.read_h5ad(INPUT_PATH)

meta = pd.read_csv(f'{META_DIR_PATH}/DNAm_samples.csv', index_col='Basename')
pheno = pd.read_csv(f'{META_DIR_PATH}/phenotypes_and_prevalent_disease.csv', index_col='Basename')
survival = pd.read_csv(f'{META_DIR_PATH}/survival.csv', index_col='Basename')
clock_results = pd.read_csv(f'{META_DIR_PATH}/DNAmAge_output.csv', index_col='Basename')
clock_sites = pd.read_csv('../resources/AllGS_CpGs.csv', index_col='CpGs')

print('Dropping NaNs...')
preprocess_func.drop_nans(amdata)

print('Merging the metadata...')
preprocess_func.merge_meta(amdata, 
                    meta, pheno, survival, clock_results, clock_sites)

print('Converting to beta values (it might take a while)...')
preprocess_func.convert_mvalues_to_betavalues(amdata)

site_index = amdata.obs.index[0]
site_data = amdata[site_index].X.flatten()
site_target = amdata.var.age
r2_score(site_data, site_target)
r_regression(amdata_T[:500], site_target)
amdata_T = amdata.copy().transpose()
