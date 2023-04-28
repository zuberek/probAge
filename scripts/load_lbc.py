# %%
import pandas as pd
import numpy as np
import anndata as ad

from src import preprocess_func

lbc_meta = pd.read_csv('/disk/scratch/methylation/LBC/lbc_meta2.csv', index_col='Basename')

X = np.load('/disk/scratch/methylation/lbc_full.npy')
X = X.T
longitudial = lbc_meta.set_index('ID').loc[lbc_meta.ID.value_counts()>1].sort_values(['ID', 'WAVE']).reset_index()

longitudial[longitudial.WAVE==3].age.hist()


cols = pd.read_csv('/disk/scratch/methylation/out.csv')
pd.Series(cols.columns.str.split('" "')[0]).to_csv('lbc_columns.csv')


columns= pd.read_csv('/disk/scratch/methylation/LBC/lbc_full.csv', nrows=0, delimiter=' ').columns.tolist()
indexes = pd.read_csv('/disk/scratch/methylation/LBC/lbc_full.csv', usecols=[0], delimiter=' ', index_col=False)
columns = pd.read_csv('lbc_columns.csv', index_col=1)
indexes = pd.read_csv('lbc_indexes.csv', index_col=1)

indexes = pd.Index(indexes.values.flatten())
indexes = [index[0] for index in indexes]
indexes = np.array(indexes)
pd.Series(indexes).to_csv('lbc_indexes.csv')
indexes = pd.read_csv('lbc_indexes.csv', index_col=1)
indexes = indexes.drop(columns=['Unnamed: 0'])
columns = pd.read_csv('lbc_columns.csv', index_col=1)
columns = columns.drop(columns=['Unnamed: 0'])

amdata = ad.AnnData(X=amdata.X, var=indexes, obs=columns)
lbc_meta = lbc_meta.drop(columns=['Unnamed: 0', 'array','pos','plate','date', 'cg16867657', 'cg05575921'])
lbc_meta.columns
amdata.var[lbc_meta.columns] = lbc_meta

amdata.var = amdata.var[['ID', 'cohort', 'WAVE', 'age', 'sex', 'BMI', 'Smoking', 'Alcohol', 'Education','set',  'DNAmAge', 'DNAmAgeHannum', 'DNAmPhenoAge', 'DNAmGrimAge',
       'DNAmTL',  
       'Total_cholesterol', 'HDL_cholesterol', 'LDL_cholesterol',
       'Total_HDL_Ratio', 'Waist_to_Hip_Ratio', 'Percent_BodyFat',
       'DunedinPOA', 'CD8T', 'CD4T', 'NK', 'Bcell', 'Mono',
       'Gran',]]

dates = pd.to_datetime(lbc_meta.date, format="%d_%m_%Y", errors="coerce").fillna(
    pd.to_datetime(lbc_meta.date, errors="coerce", format="%d/%m/%Y"))
amdata.var['date']=dates

amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)

amdata = preprocess_func.drop_nans(amdata)

indexes.to_csv('../exports/lbc_indexes.csv')
amdata.write_h5ad('../exports/lbc.h5ad')

