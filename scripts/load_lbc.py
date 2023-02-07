import pandas as pd
import numpy as np

from src.utils.plot import col

lbc_meta = pd.read_csv('/disk/scratch/methylation/LBC/lbc_meta2.csv', index_col=0)
X = np.load('/disk/scratch/methylation/lbc_full.npy')

longitudial = lbc_meta.set_index('ID').loc[lbc_meta.ID.value_counts()>1].sort_values(['ID', 'WAVE']).reset_index()

longitudial[longitudial.WAVE==3].age.hist()

indexes.loc[longitudial.Basename].head(20)

lbc_meta.cohort.hist()
lbc_meta.age.hist()

lbc_meta.reset_index()['']
lbc_meta['pos'] = lbc_meta.reset_index()
lbc_meta.set_index('Basename').index.intersection(indexes.rename(columns={'cg00000957':'index'}).set_index('index').index)


X2 = pd.read_csv('/disk/scratch/methylation/lbc_full.csv', delimiter=' ')
small = pd.read_csv('/disk/scratch/methylation/LBC/lbc_small.csv', delimiter=' ')
big = pd.read_csv('/disk/scratch/methylation/LBC/lbc_full.csv', nrows=10, delimiter=' ')

columns= pd.read_csv('/disk/scratch/methylation/LBC/lbc_full.csv', nrows=0, delimiter=' ').columns.tolist()
indexes = pd.read_csv('/disk/scratch/methylation/LBC/lbc_full.csv', usecols=[0], delimiter=' ', index_col=False)

indexes.to_csv('../exports/lbc_indexes.csv')

columns= pd.read_csv('/disk/scratch/methylation/LBC/lbc_small.csv', nrows=0, delimiter=' ').columns.tolist()
indexes = pd.read_csv('/disk/scratch/methylation/LBC/lbc_small.csv', usecols=[0], delimiter=' ', index_col=False)

indexes.rename(columns={'cg00000957':'index'}).set_index('index')

pd.read_csv(file_name, nrows=0).columns.tolist()
X.shape
lbc_meta.SampleID.unique().shape

lbc_meta
indexes = indexes.reset_index().rename(columns={'cg00000957':'Basename'}).set_index('Basename')
intersection = lbc_meta.set_index('Basename').index.intersection(indexes.index)
X_intersection = X[indexes.loc[intersection]['index']]
X_intersection.shape
indexes.index.unique()

ids = pd.DataFrame(indexes.reset_index().Basename.str.split('_', expand=True).values[:,0],columns=['ID'])
ids.ID.unique().shape

lbc_meta