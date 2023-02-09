# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.limitations_src import bootstrap_prop_fit, bootstrap_size_fit, train_test_split_indx
from src import istarmap
import concurrent.futures as cf

from itertools import repeat
import random

from functools import partial

import pickle

n_cores = 5
n_iterations = 10

data_path = '../exports/wave3_meta.h5ad'

# COHORT SIZE
cohort_size_list = [100, 200, 500, 1_000, 2_000]

bootstrap_size_partial = partial(bootstrap_size_fit,
                        data_path=data_path, )

# Create list of coordinates
comb_list = [(size, iter) 
                for size in cohort_size_list
                        for iter in range(n_iterations)]
random.seed(1)

#randomly shuffle list of coordinates to avoid high memory usage
random.shuffle(comb_list)

# Create list of coordinates
comb_list = [(size, iter) 
                for size in cohort_size_list
                        for iter in range(n_iterations)]

with Pool(processes=n_cores, maxtasksperchild=1) as p:
    results = list(tqdm(
                    p.istarmap(bootstrap_size_partial,
                                comb_list
                                ),
                    total=len(comb_list)))

with open('../exports/train_size.pk', 'wb') as f:
    pickle.dump(results, f)
    
# TOBACCO PROPORTION
prop_smoke_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


bootstrap_prop_partial = partial(bootstrap_prop_fit,
                        data_path=data_path, 
                        prop_test=0.2,
                        train_size = 700,
                        smoke_threshold=0.2)

# Create list of coordinates
comb_list = [(prop, iter) 
                for prop in prop_smoke_list
                        for iter in range(n_iterations)]

with Pool(processes=n_cores, maxtasksperchild=1) as p:
    results = list(tqdm(
                    p.istarmap(bootstrap_prop_partial,
                                comb_list
                                ),
                    total=len(comb_list)))

with open('../exports/tobacco_prop.pk', 'wb') as f:
    pickle.dump(results, f)