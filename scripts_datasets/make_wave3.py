# %% ########################
# IMPORTING
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

# %% ########################
# LOAD DATA

DATASET_NAME = 'wave3'

pheno = pd.read_csv(f'{paths.DATA_RAW_DIR}/wave3/phenotypes_and_prevalent_disease.csv', index_col='Basename')
survival = pd.read_csv(f'{paths.DATA_RAW_DIR}/wave3/survival.csv', index_col='Basename')
clock_results = pd.read_csv(f'{paths.DATA_RAW_DIR}/wave3/DNAmAge_output.csv', index_col='Basename')

# %% ########################
# ADD PARTICIPANT META DATA

# Create weighted_smoke phenotype
# Normalize pack_years data
pheno['norm_pack_years'] = np.log(1+pheno.pack_years)

# Combine ever_smoke with pack_years
pheno['weighted_smoke'] = pheno['norm_pack_years']/np.exp(pheno['ever_smoke'])

pheno['log_bmi'] = np.log(pheno.bmi)
pheno['log_pack_1'] = np.log(pheno.pack_years+1)
pheno['log_units_1'] = np.log(pheno.units+1)

# Add accelerations given to genscot participant by other clocks
####################
# rename columns to some manageable naming 
clock_results = clock_results.rename(columns={
    'AgeAccelerationResidualHannum': 'Hannum',
    'EEAA': 'Horvath',
    'AgeAccelGrim': 'GrimAge',
    'AgeAccelPheno': 'PhenoAge',
    })
clock_columns = ['Hannum','Horvath','GrimAge','PhenoAge']
pheno[clock_columns] = clock_results[clock_columns]

###
pheno[['Event', 'tte']] = survival[['Event', 'tte']]

# %% ########################
# SAVE RESULTS

pheno.to_csv(f'{paths.DATA_PROCESSED_DIR}/{DATASET_NAME}_participants.csv')

# %%
