# %% ########################
# IMPORTING
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from sklearn.preprocessing import scale

import statsmodels.formula.api as smf

wave1_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave1_participants.csv', index_col='Basename')
wave3_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave3_participants.csv', index_col='Basename')
wave4_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave4_participants.csv', index_col='Basename')

wave4_participants.acc_wave4.describe()
wave3_participants.acc_wave4.describe()


# %% ########################
# ASSOCIATIONS

# acc: smid, alco, education, glucose, cholesterol, smoking
# bias: alco, cholesterol, creatinine4

smf.ols("scale(weighted_smoke) ~  \
scale(acc_wave4) + scale(bias_wave4) + age + sex", 
wave1_participants).fit().summary()


# %%
