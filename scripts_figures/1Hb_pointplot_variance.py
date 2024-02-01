%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

amdata = ad.read_h5ad('../exports/wave3_all_fitted.h5ad')
amdata.var['age_bin'] = pd.cut(amdata.var.age, 5).astype('str')

age_bins = np.sort(amdata.var.age_bin.unique())

var_df = pd.DataFrame()
for age_bin in age_bins:
    bin = pd.DataFrame(amdata[:,amdata.var.age_bin == age_bin].X.var(axis=1).tolist(), columns=['variance'])
    bin['bin'] = age_bin
    var_df= pd.concat((var_df, bin), axis=0)
   
ax=sns.pointplot(data=var_df, x="bin", y="variance")

plot.save(ax, 'pointplot_variance_bins', format='svg')
plot.save(ax, 'pointplot_variance_bins', format='png')
