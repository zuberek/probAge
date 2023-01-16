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
                iterable=amdata, 
                ), 
            total=amdata.n_obs))

modelling.comparison_postprocess(results, amdata)

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling.fit_person,
                iterable=[amdata[:,i] for i in amdata.var.index], 
                chunksize=1,
                ), 
            total=amdata.n_vars))

