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
from src.limitations_src import non_to_uniform_sampling
import time
data_path = '../exports/wave3_meta.h5ad'


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
import gc

n_cores = 5
n_iterations = 10
MAX_TRAIN_SIZE = 1_000
np.random.seed(0)

from src.limitations_src import non_to_uniform_sampling
import time
data_path = '../exports/wave3_meta.h5ad'
train_size = 500
prop_test = 0.25
smoke_threshold = 0.2


prop_smoke = 0.1

max_iter = 20*train_size

amdata = ad.read_h5ad(data_path, 'r')
# Create weighted_smoke phenotype
# Normalize pack_years data
amdata.var['norm_pack_years'] = np.log(1+amdata.var.pack_years)

# Combine ever_smoke with pack_years
amdata.var['weighted_smoke'] = amdata.var['norm_pack_years']/np.exp(amdata.var['ever_smoke'])
amdata.var['true_smoke'] = amdata.var['weighted_smoke'] > smoke_threshold

# Proportion of data used for testing
n_participants = amdata.shape[1]
test_size = int(n_participants*prop_test)

# Create training and test list of participants
train_part_pool, test_part = train_test_split_indx(amdata, test_size)


# Split train part into smokers and non-smokers
smoker_idx_set = set(np.argwhere(np.array(amdata.var['true_smoke']) == True).flatten())
train_smokers_pool = list(train_part_pool.intersection(smoker_idx_set))
train_non_smokers_pool = list(train_part_pool - smoker_idx_set)

amdata[:, amdata.var.true_smoke == True]
amdata_train.var
amdata_smoke = amdata[:, ]
train_smoke_pool = 

amdata.var.iloc[list(train_part_pool)]


np.random.seed(int(time.time()*1_000_000%100_000))
n_smokers = int(train_part_pool*prop_smoke)
n_non_smokers = train_size-n_smokers



# subsample training set with fixed size
np.random.seed(int(time.time()*1_000_000%100_000))
subset_train_part = non_to_uniform_sampling(amdata, list(train_part_pool), train_size)

train_data = amdata[:, subset_train_part].to_memory()

a = bootstrap_size_fit(50, 0, data_path)


train_size = 100
prop_test = 0.25


max_iter = 20*train_size


amdata = ad.read_h5ad(data_path, 'r')
# Create weighted_smoke phenotype
# Normalize pack_years data
amdata.var['norm_pack_years'] = np.log(1+amdata.var.pack_years)

# Combine ever_smoke with pack_years
amdata.var['weighted_smoke'] = amdata.var['norm_pack_years']/np.exp(amdata.var['ever_smoke'])

























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


# COHORT SIZE
cohort_size_list = [100, 200, 500, 1_000, 2_000]

bootstrap_size_partial = partial(bootstrap_size_fit,
                        data_path=data_path, 
                        train_part=list(train_part),
                        test_part=test_part)

# Create list of coordinates
comb_list = [(size, iter) 
                for size in cohort_size_list
                        for iter in range(n_iterations)]
random.seed(12)

#randomly shuffle list of coordinates to avoid high memory usage
random.shuffle(comb_list)

cohort_size_list = [100]
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
np.mean([r['tobacco'] for r in results])
with open('../exports/train_size.pk', 'wb') as f:
    pickle.dump(results, f)
    
# TOBACCO PROPORTION

# Split train part into smokers and non-smokers
smoker_idx_set = set(np.argwhere(np.array(amdata.var['weighted_smoke'])>0.25).flatten())
train_smokers = list(train_part.intersection(smoker_idx_set))
train_non_smokers = list(train_part - smoker_idx_set)

# Set fitting parameters
# Cohort parameters
prop_smoke_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# prop_smoke_list = [0, 0.25, 0.5, 0.75, 1]
train_data_size = len(train_smokers)/prop_smoke_list[-1]
train_data_size = np.min([int(train_data_size), MAX_TRAIN_SIZE])

# linear fitting parameters

bootstrap_prop_partial = partial(bootstrap_prop_fit, data_path=data_path, 
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
                    p.istarmap(bootstrap_prop_partial,
                                comb_list
                                ),
                    total=len(comb_list)))

with open('../exports/tobacco_prop.pk', 'wb') as f:
    pickle.dump(results, f)

