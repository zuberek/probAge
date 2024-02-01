import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import pickle


with open ('../exports/tobacco_prop.pk', 'rb') as f:
    tobacco_results = pickle.load(f)

tob_prop_params = ['prop_smoke', 'bstrp_idx', 'mse', 'tobacco', 'bmi', 'alpha']

tobacco_dict = dict()
for key in tob_prop_params:
    tobacco_dict[key] = [d[key] for d in tobacco_results]

tobacco_dict['n_sites'] = [d['sites'].shape[0] for d in tobacco_results]

tobacco_df = pd.DataFrame(tobacco_dict)


with open ('../exports/train_size.pk', 'rb') as f:
    train_size_results = pickle.load(f)

train_size_params = ['train_size', 'bstrp_idx', 'mse', 'tobacco', 'bmi', 'alpha']

train_size_dict = dict()
for key in train_size_params:
    train_size_dict[key] = [d[key] for d in train_size_results]

train_size_dict
train_size_dict['n_sites'] = [d['sites'].shape[0] for d in train_size_results]

train_size_df = pd.DataFrame(train_size_dict)


sns.pointplot(train_size_df, x='train_size', y='tobacco')

# Comparison plots
full_cohort_association = train_size_df[train_size_df.train_size == 2_000].mean()['tobacco']

sns.pointplot(tobacco_df, x='prop_smoke', y='tobacco',color='tab:blue', label='synthetic cohort')
# sns.pointplot(tobacco_df, x='prop_smoke', y='alpha')

plt.hlines(full_cohort_association, xmin=0, xmax=8, linestyle='--', label='full cohort', color='tab:orange')
plt.legend()
plt.show()
plt.savefig('../results/prop_smoke_association.png')
plt.savefig('../results/prop_smoke_association.svg')

# 
real_association = tobacco_df[tobacco_df.prop_smoke == 0].mean()['tobacco']
sns.pointplot(train_size_df, x='train_size', y='tobacco', color='tab:orange', label='synthetic cohort')

plt.hlines(real_association, xmin=0, xmax=5, linestyle='--', label='unbiased cohort', color='tab:blue')
plt.legend()
plt.show()
plt.savefig('../results/train_size_association.png')
plt.savefig('../results/train_size_association.svg')



