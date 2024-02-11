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
downsyndrome_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/downsyndrome_participants.csv', index_col='Basename')


#########################
# ASSOCIATIONS WAVE 3

# acc: smid, alco, education, glucose, cholesterol, smoking
# bias: alco, cholesterol, creatinine4

###
# %% SMOKING

smf.ols("scale(weighted_smoke) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave1_participants).fit().summary()

smf.ols("scale(weighted_smoke) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave3_participants).fit().summary()

smf.ols("scale(weighted_smoke) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave4_participants).fit().summary()

###
# BMI

smf.ols("scale(bmi) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave1_participants).fit().summary()

smf.ols("scale(bmi) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave3_participants).fit().summary()

smf.ols("scale(bmi) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave4_participants).fit().summary()

###
# simd

smf.ols("scale(simd) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave1_participants).fit().summary()

smf.ols("scale(simd) ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
wave3_participants).fit().summary()

smf.ols("scale(simd) ~  \
scale(acc_wave4) + scale(bias_wave4) + age + sex", 
wave4_participants).fit().summary()


###
# DOWNSYDROME

smf.ols("status_binary ~  \
scale(acc_wave3) + scale(bias_wave3) + age + sex", 
downsyndrome_participants).fit().summary()

smf.ols("status_binary ~  \
acc_wave3 + sex", 
downsyndrome_participants).fit().summary()
# %%

g = sns.JointGrid()
sns.scatterplot(data=downsyndrome_participants, x='acc_wave3', y='bias_wave3', hue='status', ax=g.ax_joint)
# sns.kdeplot(data=downsyndrome_participants, x='acc_wave3', y='bias_wave3', hue='status', ax=g.ax_joint)
sns.kdeplot(data=downsyndrome_participants,  x='acc_wave3', common_norm=True, legend=False, hue='status',ax=g.ax_marg_x)
sns.kdeplot(data= downsyndrome_participants, y='bias_wave3', common_norm=True, legend=False, hue='status',ax=g.ax_marg_y)
g.ax_marg_x.vlines(downsyndrome_participants.groupby('status')['acc_wave3'].mean(), ymin=0, ymax=1, colors=colors)
g.ax_marg_y.hlines(downsyndrome_participants.groupby('status')['bias_wave3'].mean(), xmin=0, xmax=20, colors=colors)


from scipy.stats import f_oneway
f_oneway(downsyndrome_participants[downsyndrome_participants.status=='healthy'].acc_wave3,
         downsyndrome_participants[downsyndrome_participants.status!='healthy'].acc_wave3)