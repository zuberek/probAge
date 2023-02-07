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
    train_age = amdata[train_ids].obs.age
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

np.random.seed(1)
def bootstrap_fit(amdata, prop_smoke, bstrp_idx, train_smokers, train_non_smokers, train_data_size, test_part, max_iter=MAX_ITER, n_alphas=N_ALPHAS):

    np.random.seed(int(time.time()*1_000_000%100_000))
    """Bootstrap r2 and tobacoo association for different cohort sizes"""
    n_smokers = int(train_data_size*prop_smoke)
    n_non_smokers = train_data_size-n_smokers

    # Create training set
    subset_train_smokers = non_to_uniform_sampling(train_smokers, n_smokers)
    subset_train_non_smokers = non_to_uniform_sampling(train_non_smokers, n_non_smokers)
    subset_train_part = list(subset_train_smokers) + list(subset_train_non_smokers)

    # Load training data in memory
    train_data = amdata[:, subset_train_part].to_memory()

    # Set train and test age observationsS
    train_age = train_data.obs.age
    test_age = amdata[test_part].obs.age
    
    # fitting the lasso model
    lasso_model = LassoCV(n_alphas=n_alphas, max_iter=max_iter, verbose=False)
    lasso_model.fit(train_data.X, train_age)
    
    # Predict on test set using chunks
    chunk_size = 100
    n_chunks = int(np.ceil(
                len(test_part)/chunk_size)) # ceil is needed for incomplete chunks

    y_pred = np.zeros(len(test_part))
    for n in range(n_chunks):
        chunk_data = amdata[test_part[n*chunk_size:(n+1)*chunk_size]]
        y_pred[n*chunk_size:(n+1)*chunk_size] = lasso_model.predict(chunk_data.X)
        
    # Predict distance from prediction to data
    pred_diff = y_pred - test_age

    # # Compute MSE
    # MSE[prop_smoke_idx, bootstrap_iter] = np.mean((pred_diff)**2)

    mse = np.mean((pred_diff)**2)

    # compute associations
    # Create association dataframe
    association_df = pd.DataFrame({'acc':pred_diff,
                #'norm_pack_years':amdata[test_part].obs.norm_pack_years,
                'weighted_smoke': amdata[test_part].obs.weighted_smoke,
                'age':amdata[test_part].obs.age,
                'sex':amdata[test_part].obs.sex,})

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