# %%
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
# %%

with open ('../exports/train_size.pk', 'rb') as f:
    train_size_results = pickle.load(f)

train_size_params = ['train_size', 'bstrp_idx', 'mse', 'tobacco', 'bmi', 'alpha']

train_size_dict = dict()
for key in train_size_params:
    train_size_dict[key] = [d[key] for d in train_size_results]

train_size_dict
train_size_dict['n_sites'] = [d['sites'].shape[0] for d in train_size_results]

train_size_df = pd.DataFrame(train_size_dict)


# %%
real_association = tobacco_df[tobacco_df.prop_smoke == 0].mean()['tobacco']

ax = plot.row(figsize=(9.5, 6))
ax2 = plt.twinx()
plot.fonts(8)
sns.despine(right=False)

sns.pointplot(train_size_df, ax=ax, x='train_size', y='tobacco', 
              color=colors[0], label='Smoking association', errorbar='ci', capsize=.3, linewidth=1)
sns.pointplot(train_size_df,x='train_size', y='mse',color=colors[1], 
              label ='MSE', ax=ax2, errorbar='ci', capsize=.3, linewidth=1)

# Solution for having two legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

ax.set_ylabel("Smoking \n association \n" + r"($\beta$-coeff)")
ax2.set_ylabel('MSE')
ax.set_xlabel('Training size (n)')
ax.get_figure().tight_layout()

# %%
ax.get_figure().savefig('../figures/fig1/1E_prop_smoke_association.png', transparent=False)
ax.get_figure().savefig('../figures/fig1/1E_prop_smoke_association.svg', transparent=False)