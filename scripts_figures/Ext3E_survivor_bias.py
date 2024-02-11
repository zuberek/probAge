# %% ########################
### LOADING

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

amdata_YOUNG = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_person_fitted_YOUNG75.h5ad')
participants = pd.read_csv('../exports/wave3_participants.csv', index_col=0)

import plotly.express as px
px.scatter(amdata_YOUNG[:,amdata_YOUNG.var.age>60].var, x='age',y='acc', marginal_y='histogram')
older_idx = amdata[:,amdata.var.age>75].var.index

diff = amdata_YOUNG[:,older_idx].var.acc - participants.loc[older_idx].acc
log_abs_diff = np.log(np.abs(diff))

# %%
 
sns.scatterplot(x=np.log(np.abs(participants.loc[older_idx].acc)), y =log_abs_diff)
plt.axline((0,0), slope=1, color='r', linestyle='--', label='x=y')
plt.ylabel('log (|acc_diff|)')
plt.xlabel('log (|acc|)')
plt.legend()
 
# %%
ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.despine()

mean_omd = np.mean(log_abs_diff - np.log(np.abs(participants.loc[older_idx].acc)))
mean_omd = np.round(mean_omd, 1)
# g=sns.displot(x=log_abs_diff - np.log(np.abs(participants.loc[older_idx].acc)), color=colors[0], kind='kde', rug=True)
sns.kdeplot(ax=ax, x=log_abs_diff - np.log(np.abs(participants.loc[older_idx].acc)))
sns.rugplot(ax=ax, x=log_abs_diff - np.log(np.abs(participants.loc[older_idx].acc)), color=colors[0])
plt.axvline(x=mean_omd, color=colors[1], linestyle='--')
plt.xlabel('Order of magnitude difference:\n log (|acc_diff|) - log (|acc|)')
 
plt.xticks([-6,-4,-2, 0,  2] + [mean_omd])
plt.gca().get_xticklabels()[-1].set_color(colors[1])
ax.get_figure().tight_layout()
# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3E_survivor_bias.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3E_survivor_bias.svg')

# %%

