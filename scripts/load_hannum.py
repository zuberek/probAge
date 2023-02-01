import pandas as pd
import anndata as ad

DATA_PATH = '../../methylation/GSE40279_average_beta.txt'
meta_path = '../resources/hannum_meta.csv'

hannum_meta = pd.read_csv(meta_path, index_col=0)

hannum = pd.read_csv(DATA_PATH)
hannum = ad.read_text(DATA_PATH)

hannum.var['age'] = hannum_meta['Age']
hannum.write_h5ad('../exports/hannum.h5ad')