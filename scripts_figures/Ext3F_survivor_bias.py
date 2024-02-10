# %% ########################
### LOADING

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

amdata_YOUNG = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_person_fitted_YOUNG75.h5ad')
participants = pd.read_csv('../exports/wave3_participants.csv', index_col='Basename')

import plotly.express as px
px.scatter(amdata_YOUNG[:,amdata_YOUNG.var.age>60].var, x='age',y='acc', marginal_y='histogram')
older_idx = amdata[:,amdata.var.age>75].var.index

amdata_YOUNG.var
amdata_YOUNG.var.age.hist()

diff = amdata_YOUNG[:,older_idx].var.acc - participants.loc[older_idx].acc
log_abs_diff = np.log(np.abs(diff))

# %%
 
sns.scatterplot(x=np.log(np.abs(participants.loc[older_idx].acc)), y =log_abs_diff)
plt.axline((0,0), slope=1, color='r', linestyle='--', label='x=y')
plt.ylabel('log (|acc_diff|)')
plt.xlabel('log (|acc|)')
plt.legend()
 
# %%
 
mean_omd = np.mean(log_abs_diff - np.log(np.abs(participants.loc[older_idx].acc)))
ax=sns.displot(x=log_abs_diff - np.log(np.abs(participants.loc[older_idx].acc)), color=colors[0], kind='kde', rug=True)
plt.axvline(x=mean_omd, color=colors[1], linestyle='--')
plt.xlabel('Order of magnitude difference:\n log (|acc_diff|) - log (|acc|)')
 
plt.xticks([-6,-4,-2, 0,  2] + [-1.2])
plt.gca().get_xticklabels()[-1].set_color(colors[1])
# %%

