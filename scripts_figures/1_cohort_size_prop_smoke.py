import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
import pickle

import anndata as ad
data_path = '../exports/wave3_meta.h5ad'
amdata = ad.read_h5ad('../exports/wave3_meta.h5ad', 'r')

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

# Comparison plots
full_cohort_association = train_size_df[train_size_df.train_size == 2_000].mean()['tobacco']

sns.pointplot(tobacco_df, x='prop_smoke', y='tobacco',color='tab:blue', label='synthetic cohort')
plt.legend()
ax2 = plt.twinx()
sns.pointplot(tobacco_df,x='prop_smoke', y='mse', color='tab:orange', label ='mse', ax=ax2)
plt.legend()
plt.savefig('../results/prop_smoke_association.png')
plt.savefig('../results/prop_smoke_association.svg')

# 
real_association = tobacco_df[tobacco_df.prop_smoke == 0].mean()['tobacco']

sns.pointplot(train_size_df, x='train_size', y='tobacco', color='tab:blue', label='synthetic cohort')
plt.legend()
ax2 = plt.twinx()
sns.pointplot(train_size_df,x='train_size', y='mse',color='tab:orange', label ='mse', ax=ax2)
plt.legend()
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig('../results/train_size_association.png')
plt.savefig('../results/train_size_association.svg')

jan_site = 'cg05575921'


indexes = amdata.obs.iloc[a].index

for r in tobacco_results:
    indexes = amdata.obs.iloc[r['sites']].index.to_list()
    if jan_site in indexes:
        print(r['prop_smoke'])




for r in train_size_results:
    indexes = amdata.obs.iloc[r['sites']].index.to_list()
    if jan_site in indexes:
        print(r['train_size'])


r = train_size_results[0]
amdata.obs.iloc[r['sites']].index.to_list()

if jan_site in ['hola', 'no']:
    print('yes')


a = np.array([site for r in train_size_results for site in r['sites']])
indexes = np.array(amdata.obs.iloc[a].index)

np.argwhere(indexes ==jan_site)