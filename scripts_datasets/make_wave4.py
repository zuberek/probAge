# %% ########################
# IMPORTING
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

# %% ########################
# LOAD DATA

DIR_PATH = '/mnt/tchandra-lab/EJYang/methylclock/GS_wave4'
pheno = pd.read_csv(f'{DIR_PATH}/2023-08-02_w4_phenotypes.csv', index_col='id')
survival = pd.read_csv(f'{DIR_PATH}/2023-08-02_w4_deaths.csv', index_col='id')

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

###
pheno[['dob_ym', 'dod_ym']] = survival[['dob_ym', 'dod_ym']]

# %%
pheno.to_csv('../exports/wave4_participants.csv')
