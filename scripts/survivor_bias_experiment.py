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
participant_indexes = amdata[:,amdata.var.age>75].var.index

sns.scatterplot(x=participants.loc[participant_indexes].acc_wave3, y=amdata_YOUNG[: , participant_indexes].var.acc)
sns.scatterplot(data=amdata_YOUNG[:,amdata_YOUNG.var.age>60].var, x='age',y='acc')
amdata_YOUNG.var
amdata_YOUNG.var.age.hist()

# %% ########################
### PLOT 1
participant_indexes = amdata_YOUNG[:,amdata_YOUNG.var.age>75].var.index

diff = amdata_YOUNG[:,participant_indexes].var.acc - participants.loc[participant_indexes].acc
g = sns.JointGrid()
sns.scatterplot(data=amdata.var, alpha=0.5, x='acc', y='bias', color='tab:grey', ax=g.ax_joint)
sns.kdeplot(data=amdata.var, x='acc', y='bias', color='black',  ax=g.ax_joint)
sns.kdeplot(data=amdata.var, x='acc', color='black', ax=g.ax_marg_x, legend=False)
sns.kdeplot(data=amdata.var, y='bias', color='black', ax=g.ax_marg_y, legend=False)

# %% ########################
### PLOT 1
g = sns.JointGrid()
sns.scatterplot(ax=g.ax_joint, x=participants.loc[participant_indexes].age, y=diff, label='survivor effect')
# sns.scatterplot(ax=g.ax_joint, x=participants.loc[participant_indexes].age, y=participants.loc[participant_indexes].acc, label='acceleration')
sns.scatterplot(ax=g.ax_joint, x=participants.age, y=participants.acc, label='acceleration')

sns.kdeplot(ax=g.ax_marg_y, y=diff)
# sns.kdeplot(ax=g.ax_marg_y, y=participants.loc[participant_indexes].acc)
# sns.scatterplot(ax=g.ax_joint, x=participants.age, y=participants.acc, label='acceleration')
sns.kdeplot(ax=g.ax_marg_y, y=participants.acc)

# %% ########################
### PLOT 2
ax=sns.scatterplot(x=participants.loc[participant_indexes].acc, y=diff)
ax.axline((0,0), slope=1)
ax.set_xlabel('acc')
ax.set_ylabel('diff')
# %%
amdata_YOUNG.var['old'] = False
amdata_YOUNG.var.loc[participant_indexes, 'old'] = True
sns.scatterplot(x= amdata_YOUNG[:,participant_indexes].var.age, y=amdata_YOUNG[:,participant_indexes].var.ace)
sns.scatterplot(data=amdata_YOUNG.var, x= 'age', y='acc', hue='old')
ax=sns.scatterplot(y=amdata_YOUNG.var.acc, x=participants.acc, hue=amdata_YOUNG.var.old)
ax.axline((0,0), slope=1)
ax.set_xlabel('acc')
ax.set_ylabel('diff')