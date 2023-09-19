# %% ########################
# IMPORTING
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from sklearn.preprocessing import scale
from src import paths

import statsmodels.formula.api as smf

dataset = 'wave1'
wave1 = amdata_src.AnnMethylData(f'wave1_100external.h5ad', backed='r')
wave3 = amdata_src.AnnMethylData(f'wave3_100external.h5ad', backed='r')
wave4 = amdata_src.AnnMethylData(f'wave4_100external.h5ad', backed='r')

wave3_MAP = amdata_src.AnnMethylData(f'../exports/wave3_MAP_acc.h5ad', backed='r')
wave3.var['acc_old'] = wave3_MAP.var.acc
wave3.var['bias_old'] = wave3_MAP.var.bias

wave4_participants = pd.read_csv('../scripts_datasets/wave4_participants.csv', index_col=0)

wave4.var['bmi'] = wave4_participants['bmi']
wave4.var['pack_years'] = wave4_participants['pack_years']
wave4.var['units'] = wave4_participants['units']
wave4.var['weighted_smoke'] = wave4_participants['weighted_smoke']

# %% ########################
# IMPORTING
ax1,ax2 = plot.row(['Acc', 'Bias'], 'Wave 4 Results')
sns.scatterplot(data=wave4.var, x='acc_wave4', y='acc_ewas', ax=ax1)
sns.lineplot(x=[0.5,2], y=[0.5,2], ax=ax1)

sns.scatterplot(data=wave4.var, x='bias_wave4', y='bias_ewas', ax=ax2)
sns.lineplot(x=[-0.05,0.05], y=[-0.05,0.05], ax=ax2)

# %% ########################
# PREPROCESS
wave1.var['log_bmi'] = np.log(wave1.var.bmi)
wave1.var['log_pack_1'] = np.log(wave1.var.pack_years+1)
wave1.var['log_units_1'] = np.log(wave1.var.units+1)

wave3.var['log_bmi'] = np.log(wave3.var.bmi)
wave3.var['log_pack_1'] = np.log(wave3.var.pack_years+1)
wave3.var['log_units_1'] = np.log(wave3.var.units+1)

wave4.var['log_bmi'] = np.log(wave4.var.bmi)
wave4.var['log_pack_1'] = np.log(wave4.var.pack_years+1)
wave4.var['log_units_1'] = np.log(wave4.var.units+1)





# %% ########################
# ASSOCIATIONS

# acc: smid, alco, education, glucose, cholesterol, smoking
# bias: alco, cholesterol, creatinine4

smf.ols("scale(weighted_smoke) ~ scale(acc_wave4)  \
+ scale(bias_wave4) + age + sex", wave4.var).fit().summary()

smf.ols("scale(weighted_smoke) ~ scale(acc_ewas)  \
+ scale(bias_ewas) + age + sex", wave4.var).fit().summary()

smf.ols("scale(weighted_smoke) ~ scale(acc_old)  \
+ scale(bias_old) + age + sex", wave3.var).fit().summary()







# %% ########################
# ASSOCIATIONS
wave4 = amdata_src.AnnMethylData(f'../exports/wave4_meta.h5ad', backed='r')

wave4_ewas= pd.read_csv('../scripts_master/wave4_ewas_fits.csv', index_col=['site','model','param'])
wave4_wave4 = pd.read_csv('../scripts_master/wave4_wave4_fits.csv', index_col=['site','model','param'])

wave4 = wave4[:,:100]
wave4 = wave4.to_memory()

param_list = wave4_wave4.xs('bio', level='model').index.get_level_values(level='param')
for param in param_list:
    wave4.var[f'{param}_wave4'] = wave4_wave4.loc[(slice(None),'bio', param)]['mean'].values
    wave4.var[f'{param}_ewas'] = wave4_ewas.loc[(slice(None),'bio', param)]['mean'].values

wave4.write_h5ad('wave4_100external.h5ad')
# %%
