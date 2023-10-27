# %% ########################
# Import packages
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

jobname = "wave3"

import os
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import time
import pickle


# %% ########################
# Loading data
if jobname == "wave3":
    amdata = amdata_src.AnnMethylData('/exports/igmm/eddie/tchandra-lab/EJYang/methylclock/GS_mval/wave3_meta.h5ad', backed='r')
elif jobname == "wave4":
    amdata = amdata_src.AnnMethylData('exports/wave4/wave4_meta.h5ad', backed='r')
else:
    # Handle the case when job name is neither "ewas" nor "wave4"
    raise ValueError("Invalid job name")

# select the top 10_000 sites by r2 value
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).head(10_000).index]
amdata = amdata.to_memory()

# Replace 0 and 1
amdata.X = np.where(amdata.X == 0, 0.0001, amdata.X)
amdata.X = np.where(amdata.X == 1, 0.9999, amdata.X)

sns.scatterplot(x=amdata.var.age, y =amdata.X[0].flatten())

# %%
sns.scatterplot(x=y_train, y=y_train_pred, label='train')
sns.scatterplot(x=y_test, y=y_test_pred, label='test')
plt.xlabel('Age')
plt.xlabel('Predicted Age')

# %%
    # Proportion of data used for testing

# %% ########################
# some bits and bulbs to test the script
def train_test_split_indx(amdata, test_size):
    n_participants = amdata.shape[1]
    # Create training and test list of participants
    test_part = np.random.choice(n_participants, size=test_size, replace=False)
    train_part = set([i for i in np.arange(0, n_participants) if i not in test_part])

    return train_part, test_part

def non_to_uniform_sampling(amdata, train_ids, n_samples):
    # Return empty list if no samples are to be drawn
    if n_samples == 0:
        return []
    
    # Define a uniform distribution with min and max associated to extreme observations
    train_age = amdata[:, train_ids].var.age
    train_arange = [np.min(train_age), np.max(train_age)]
    uniform_sampling = np.random.uniform(train_arange[0], train_arange[1], size=n_samples)

    # Create a pandas dataframe
    train_age = pd.DataFrame(train_age)
    train_age['abs_diff'] = 0
    train_age['selection'] = 0

    # # loop through random sampling of uniform ages
    # # find nearest element in train data
    for i in uniform_sampling:
        train_age['abs_diff'] = np.abs(train_age.age-i)
        # Find closest participant not in selection 
        idx_min = train_age[train_age.selection == 0].abs_diff.idxmin()
        train_age.at[idx_min, 'selection'] = 1
        
    selection_idx = train_age[train_age.selection==1].index
    return selection_idx
# %% ########################
# Define the model
MAX_ITER = 1_000
N_ALPHAS = 5

def lasso_wave_fit(amdata, n_alphas=N_ALPHAS, verbose=False, n_jobs=None, prop_test=0.2, train_size=700):
    """Bootstrap tobacoo association for wave3 and wave4"""
    np.random.seed(int(time.time()*1_000_000%100_000))
    max_iter = 20*train_size
    
    n_participants = amdata.shape[1]
    test_size = int(n_participants*prop_test)

    # Create training and test list of participants
    train_part_pool, test_part = train_test_split_indx(amdata, test_size)

    # subsample training set with fixed size
    subset_train_part = non_to_uniform_sampling(amdata, list(train_part_pool), train_size)

    # Load training data in memory
    #if amdata.isbacked:
    train_data = amdata[:, subset_train_part].to_memory()
    #else:
    #    train_data = amdata[:, subset_train_part]
    train_data = train_data.T

    # Set train and test age observationsS
    train_age = train_data.obs.age
    test_age = amdata[:, test_part].var.age
    
    # fitting the lasso model
    lasso_model = LassoCV(n_alphas=n_alphas, max_iter=max_iter, verbose=verbose, n_jobs=n_jobs)
    lasso_model.fit(train_data.X, train_age)
    
    # Predict on test set using chunks
    chunk_size = 100
    n_chunks = int(np.ceil(
                len(test_part)/chunk_size)) # ceil is needed for incomplete chunks

    y_pred = np.zeros(len(test_part))
    for n in range(n_chunks):
        chunk_data = amdata[:, test_part[n*chunk_size:(n+1)*chunk_size]].to_memory().T
        y_pred[n*chunk_size:(n+1)*chunk_size] = lasso_model.predict(chunk_data.X)

    # Predict distance from prediction to data
    pred_diff = y_pred - test_age

    # Compute MSE
    mse = np.mean((pred_diff)**2)

    # compute associations
    # Create association dataframe
    association_df = pd.DataFrame({'acc':pred_diff,
                #'norm_pack_years':amdata[test_part].obs.norm_pack_years,
                'weighted_smoke': amdata[:, test_part].var.weighted_smoke,
                'bmi': np.log(amdata[:, test_part].var.bmi),
                'age':amdata[:, test_part].var.age,
                'sex':amdata[:, test_part].var.sex,})
    
    association_df = association_df.dropna()
    association_mod = smf.ols("acc ~ scale(weighted_smoke) + scale(bmi) + age + sex", association_df)
    association_fit = association_mod.fit()

    # Update stats
    # tobacco[prop_smoke_idx, bootstrap_iter] = association_fit.params['norm_pack_years']
    tobacco = association_fit.params['scale(weighted_smoke)']
    bmi = association_fit.params['scale(bmi)']

    lasso_sites = np.argwhere(lasso_model.coef_ != 0).flatten()
    results_dict = {'mse': mse,
                    'tobacco': tobacco,
                    'bmi': bmi,
                    'sites': lasso_sites,
                    'alpha': lasso_model.alpha_}

    return results_dict


# %% ########################
# Fitting the model
results = lasso_wave_fit(amdata=amdata)

# %%
wave4 = amdata_src.AnnMethylData('/exports/igmm/eddie/tchandra-lab/EJYang/methylclock/ProbAge/exports/wave4/wave4_meta.h5ad', backed='r')
# update the phenotype information for wave4 espically tobacco and bmi 
wave4_meta = pd.read_csv('/exports/igmm/eddie/tchandra-lab/EJYang/methylclock/ProbAge_result/final_results/wave4_participants.csv')

wave4 = wave4[wave4.obs.sort_values('r2', ascending=False).head(10_000).index]
wave4 = wave4.to_memory()

wave4.X = np.where(wave4.X == 0, 0.0001, wave4.X)
wave4.X = np.where(wave4.X == 1, 0.9999, wave4.X)

wave4_meta = wave4_meta.set_index('id').loc[wave4.var['id']]
wave4.var['weighted_smoke'] = wave4.var['id'].map(wave4_meta['weighted_smoke'])
wave4.var['bmi'] = wave4.var['id'].map(wave4_meta['bmi'])
results_wave4 = lasso_wave_fit(wave4)

# %% ########################
print(results)


sns.scatterplot(x=y_train, y=y_train_pred, label='train')
sns.scatterplot(x=y_test, y=y_test_pred, label='test')
plt.xlabel('Age')
plt.xlabel('Predicted Age')