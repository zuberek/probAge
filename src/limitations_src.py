import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from sklearn.linear_model import LassoCV
import statsmodels.formula.api as smf
import time
import gc

MAX_ITER = 1_000
N_ALPHAS = 5

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


def bootstrap_prop_fit(prop_smoke, bstrp_idx, data_path, prop_test=0.2, train_size=700, smoke_threshold=0.2, n_alphas=N_ALPHAS, n_jobs=None):
    """Bootstrap r2 and tobacoo association for different proportions of smokers"""

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

    n_smokers = np.min([int(train_size*prop_smoke), len(train_smokers_pool)])
    n_non_smokers = train_size-n_smokers

    # Create training set
    subset_train_smokers = non_to_uniform_sampling(amdata, train_smokers_pool, n_smokers)
    subset_train_non_smokers = non_to_uniform_sampling(amdata, train_non_smokers_pool, n_non_smokers)
    subset_train_part = list(subset_train_smokers) + list(subset_train_non_smokers)


    train_data = amdata[:, subset_train_part].to_memory()
    train_data = train_data.T

    # Set train and test age observationsS
    train_age = train_data.obs.age
    test_age = amdata[:, test_part].var.age
    
    # fitting the lasso model
    lasso_model = LassoCV(n_alphas=n_alphas, max_iter=max_iter, verbose=False, n_jobs=n_jobs)
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
                'age':amdata[:, test_part].var.age,
                'sex':amdata[:, test_part].var.sex,})
    
    association_df = association_df.dropna()
    association_mod = smf.ols("acc ~ weighted_smoke + age + sex", association_df)
    association_fit = association_mod.fit()

    # Update stats
    # tobacco[prop_smoke_idx, bootstrap_iter] = association_fit.params['norm_pack_years']
    tobacco = association_fit.params['weighted_smoke']

    lasso_sites = np.argwhere(lasso_model.coef_ != 0).flatten()
    results_dict = {'prop_smoke':prop_smoke, 
                    'bstrp_idx': bstrp_idx,
                    'mse': mse,
                    'tobacco': tobacco,
                    'sites': lasso_sites,
                    'alpha': lasso_model.alpha_}

    return results_dict



def bootstrap_size_fit(train_size, bstrp_idx, data_path, prop_test=0.2,
                       n_alphas=N_ALPHAS, verbose=False, n_jobs=None):
    """Bootstrap tobacoo association for different cohort sizes"""
    
    max_iter = 20*train_size
    
    amdata = ad.read_h5ad(data_path, 'r')
    # Create weighted_smoke phenotype
    # Normalize pack_years data
    amdata.var['norm_pack_years'] = np.log(1+amdata.var.pack_years)

    # Combine ever_smoke with pack_years
    amdata.var['weighted_smoke'] = amdata.var['norm_pack_years']/np.exp(amdata.var['ever_smoke'])

    # Proportion of data used for testing
    n_participants = amdata.shape[1]
    test_size = int(n_participants*prop_test)

    # Create training and test list of participants
    train_part_pool, test_part = train_test_split_indx(amdata, test_size)

    # subsample training set with fixed size
    np.random.seed(int(time.time()*1_000_000%100_000))
    subset_train_part = non_to_uniform_sampling(amdata, list(train_part_pool), train_size)

    # Load training data in memory
    train_data = amdata[:, subset_train_part].to_memory()
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
                'age':amdata[:, test_part].var.age,
                'sex':amdata[:, test_part].var.sex,})
    
    association_df = association_df.dropna()
    association_mod = smf.ols("acc ~ weighted_smoke + age + sex", association_df)
    association_fit = association_mod.fit()

    # Update stats
    # tobacco[prop_smoke_idx, bootstrap_iter] = association_fit.params['norm_pack_years']
    tobacco = association_fit.params['weighted_smoke']

    lasso_sites = np.argwhere(lasso_model.coef_ != 0).flatten()
    results_dict = {'train_size':train_size, 
                    'bstrp_idx': bstrp_idx,
                    'mse': mse,
                    'tobacco': tobacco,
                    'sites': lasso_sites,
                    'alpha': lasso_model.alpha_}

    return results_dict


def train_test_split_indx(amdata, test_size):
    n_participants = amdata.shape[1]
    # Create training and test list of participants
    test_part = np.random.choice(n_participants, size=test_size, replace=False)
    train_part = set([i for i in np.arange(0, n_participants) if i not in test_part])

    return train_part, test_part
