%load_ext autoreload
%autoreload 2
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import preprocess_func

from sklearn.feature_selection import r_regression

INPUT_PATH = '../exports/wave3_raw.h5ad'
INPUT_PATH = '../exports/wave3_meta.h5ad'
META_DIR_PATH = '/afs/inf.ed.ac.uk/user/s17/s1768506/disk/methylation/genscot_meta/wave3'

print('Loading the dataset (it might take a while)...')
amdata = ad.read_h5ad(INPUT_PATH)

meta = pd.read_csv(f'{META_DIR_PATH}/DNAm_samples.csv', index_col='Basename')
pheno = pd.read_csv(f'{META_DIR_PATH}/phenotypes_and_prevalent_disease.csv', index_col='Basename')
survival = pd.read_csv(f'{META_DIR_PATH}/survival.csv', index_col='Basename')
clock_results = pd.read_csv(f'{META_DIR_PATH}/DNAmAge_output.csv', index_col='Basename')
clock_sites = pd.read_csv('../resources/AllGS_CpGs.csv', index_col='CpGs')

print('Dropping NaNs...')
amdata = preprocess_func.drop_nans(amdata)

print('Merging the metadata...')
preprocess_func.merge_meta(amdata, 
                    meta, pheno, survival, clock_results, clock_sites)

print('Converting to beta values (it might take a while)...')
amdata.X = preprocess_func.convert_mvalues_to_betavalues(amdata)

print('Computing the r2 (it might take a while)...')
def r2(site_index):
    return r_regression(amdata[site_index].X.T, amdata.var.age)[0]**2
r2('cg16867657')


with Pool() as p:
    output = list(tqdm(p.imap(
            func=r2, 
            iterable=amdata.obs.index,
            chunksize=len(amdata.obs.index)//cpu_count()
            ),
        total=len(amdata.obs.index)))
# [01:47<00:00, 7171.79it/s] (16 CPUs)

amdata.obs['r2'] = output

# TODO: Select top 1k for further processing
top_r2 = amdata.obs.sort_values('r2', ascending=False).index[:1000]
amdata[top_r2].write_h5ad('../exports/wave3_linear.h5ad')

(amdata.obs.r2>0.3).value_counts()


amdata.write_h5ad('../exports/wave3_meta.h5ad')

