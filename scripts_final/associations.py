# %% ########################
# IMPORTING
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from sklearn.preprocessing import scale

import statsmodels.formula.api as smf

# COL_NAMES =['acc_wave4', 'bias_wave4', 'acc_ewasKNN', 'bias_ewasKNN']

wave1_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave1_participants.csv', index_col='Basename')
wave3_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave3_participants.csv', index_col='Basename')
wave4_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave4_participants.csv', index_col='Basename')


# %% ########################
# ASSOCIATIONS WAVE 3

# acc: smid, alco, education, glucose, cholesterol, smoking
# bias: alco, cholesterol, creatinine4

smf.ols("scale(weighted_smoke) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave3_participants).fit().summary()

wave1_participants.age = np.float(wave1_participants.age)
wave1_participants.dtypes

smf.ols("scale(weighted_smoke) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave3_participants).fit().summary()

smf.ols("scale(weighted_smoke) ~  \
scale(acc_ewasKNN) + scale(bias_ewasKNN) + scale(bmi) + age + sex", 
wave3_participants).fit().summary()

sns.scatterplot(data=wave3_participants, x='acc_ewasKNN', y='acc_wave4')
sns.lineplot(x=[0,1], y=[0,1])

wave3_participants.acc_wave4.max()
wave3_participants.acc_ewasKNN.max()


# %% ########################
# ASSOCIATIONS WAVE 4

# acc: smid, alco, education, glucose, cholesterol, smoking
# bias: alco, cholesterol, creatinine4

smf.ols("scale(weighted_smoke) ~  \
scale(acc_wave4) + scale(bias_wave4) + age + sex", 
wave4_participants).fit().summary()

smf.ols("scale(weighted_smoke) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave4_participants).fit().summary()

smf.ols("scale(weighted_smoke) ~  \
scale(acc_ewasKNN) + scale(bias_ewasKNN) + age + sex", 
wave4_participants).fit().summary()

sns.scatterplot(data=wave3_participants, x='acc_wave3', y='acc_wave4')

wave3_participants.acc_wave4.max()
wave3_participants.acc_ewasKNN.max()
