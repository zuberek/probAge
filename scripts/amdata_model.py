%load_ext autoreload
%autoreload 2
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

N_SITES =  10

amdata = amdata.AnnMethylData('../exports/wave3_meta.h5ad', backed='r')
amdata = amdata[amdata.sites.sort_values('r2', ascending=False).index[:N_SITES]].to_memory()

sns.scatterplot(x= amdata.var.age, y=amdata['cg07547549'].X.flatten())

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling.fit_and_compare,
                iterable=[amdata[i] for i in amdata.obs.index], 
                ), 
            total=amdata.n_obs))

def comparison_postprocess(results, amdata):
    fits = pd.DataFrame()
    comparisons = pd.DataFrame()
    for site in results:
        fit, comparison = site
        fits = pd.concat([fits, fit])
        comparisons = pd.concat([comparisons, comparison])

    amdata.obs['mean_slope'] = fits.loc[(slice(None),'drift','mean_slope')].MAP.values
    amdata.obs['mean_inter'] = fits.loc[(slice(None),'drift','mean_inter')].MAP.values
    amdata.obs['var_slope'] = fits.loc[(slice(None),'drift','var_slope')].MAP.values
    amdata.obs['var_inter'] = fits.loc[(slice(None),'drift','var_inter')].MAP.values


fits.reset_index()['param'].unique()
np.array(results, dtype='object')
results[0][0]

results

modelling.fit_and_compare(amdata[0])


    
