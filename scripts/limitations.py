import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.limitations import non_to_uniform_sampling, bootstrap_fit
from src import istarmap
import concurrent.futures as cf
import statsmodels.formula.api as smf

from itertools import repeat
import time


from functools import partial

import pickle
import gc

N_CORES = 5
MAX_ITER = 1_000
N_ALPHAS = 5
MAX_TRAIN_SIZE = 1_000

data_path = '../exports/wave3_meta.h5ad'
amdata = ad.read_h5ad(data_path, 'r')

# Create weighted_smoke phenotype
# Normalize pack_years data
amdata.var['norm_pack_years'] = np.log(1+amdata.var.pack_years)

# Combine ever_smoke with pack_years
amdata.var['weighted_smoke'] = amdata.var['norm_pack_years']/np.exp(amdata.var['ever_smoke'])


# Proportion of data used for testing
prop_test = 0.25
n_participants = amdata.shape[1]
test_size = int(n_participants*prop_test)

# Create training and test list of participants
test_part = np.random.choice(n_participants, size=test_size, replace=False)
train_part = set([i for i in np.arange(0, n_participants) if i not in test_part])


# Split train part into smokers and non-smokers
smoker_idx_set = set(np.argwhere(np.array(amdata.var['weighted_smoke'])>0.2).flatten())
train_smokers = list(train_part.intersection(smoker_idx_set))
train_non_smokers = list(train_part - smoker_idx_set)

# Set fitting parameters
# Cohort parameters
# prop_smoke_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
prop_smoke_list = [0, 0.25, 0.5, 0.75, 1]
train_data_size = len(train_smokers)/prop_smoke_list[-1]
train_data_size = np.min([int(train_data_size), MAX_TRAIN_SIZE])

# linear fitting parameters

n_cores = 5
n_iterations = 10

bootstrap_fit_partial = partial(bootstrap_fit, data_path=data_path, 
                        train_data_size=train_data_size,
                        train_smokers=train_smokers,
                        train_non_smokers=train_non_smokers,
                        test_part=test_part)


# Create list of coordinates
comb_list = [(prop, iter) 
                for prop in prop_smoke_list
                        for iter in range(n_iterations)]

with Pool(processes=n_cores, maxtasksperchild=1) as p:
    results = list(tqdm(
                    p.istarmap(bootstrap_fit_partial,
                                comb_list
                                ),
                    total=len(comb_list)))

with open('../exports/tobacco_prop.npy', 'wb') as f:
    pickle.dump(results, f)