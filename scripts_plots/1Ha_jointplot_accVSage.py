# %% ########################
# LOADING

# %load_ext autoreload 
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from sklearn.linear_model import LinearRegression

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_meta.h5ad', backed='r')
wave3_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave3_participants.csv', index_col='Basename')
amdata.var.age.
N_BINS = 5

participants = wave3_participants

participants['age_bin'] = pd.cut(amdata.var.age, N_BINS, precision=0).astype('str')

age_bins = np.sort(participants.age_bin.unique())

bin_size = participants.groupby('age_bin')['Sample_Name'].count()
bin_centers = participants.groupby('age_bin')['age'].mean().values[:,np.newaxis]

var_df = pd.DataFrame()
for age_bin in tqdm(age_bins):
    bin = pd.DataFrame(amdata[:,participants.age_bin == age_bin].X.var(axis=1).tolist(), columns=['variance'])
    bin['bin'] = age_bin
    var_df= pd.concat((var_df, bin), axis=0)


# %%  
# Run for a our acceleration
CLOCK = 'acc_wave3'

bin_means = participants.groupby('age_bin')[CLOCK].mean().values
bin_sds = participants.groupby('age_bin')[CLOCK].std().values
mean_fit = LinearRegression().fit(X = bin_centers,  y= bin_means)
sd_fit = LinearRegression().fit(X = bin_centers,  y= bin_sds)

mean = mean_fit.predict(bin_centers)
std = sd_fit.predict(bin_centers)
std_up = mean + 2*std
std_down = mean - 2*std


# %%  
# g = sns.JointGrid(ylim=(-20,20))
g = sns.JointGrid()
sns.scatterplot(ax=g.ax_joint, data=participants, x='age', y=CLOCK, alpha=0.3, hue='age_bin', 
                palette=CON_PALLETE,   hue_order=age_bins, legend=False)
sns.lineplot(ax=g.ax_joint, x=bin_centers.flatten(), color='tab:blue', label='Mean', y=mean)
sns.lineplot(ax=g.ax_joint, x=bin_centers.flatten(), color='tab:blue', label='2*Standard deviation', y=std_up)
sns.lineplot(ax=g.ax_joint, x=bin_centers.flatten(), color='tab:blue', y=std_down)


sns.boxplot(data=participants, y=CLOCK, x=participants.age_bin, hue_order=age_bins, 
                ax=g.ax_marg_y, palette=CON_PALLETE, showfliers=False)
# sns.pointplot(data=var_df, x="bin", y="variance", ax=g.ax_marg_x)
g.refline(y=0)
g.fig.subplots_adjust(top=0.7)

# %%  
# # g = sns.JointGrid(ylim=(-20,20))
# sns.scatterplot(ax=g.ax_joint, data=participants, x='age', y=CLOCK, alpha=0.3, hue='age_bin', 
#                 palette=CON_PALLETE,   hue_order=age_bins, legend=False)
# sns.lineplot(ax=g.ax_joint, x=bin_centers.flatten(), color='tab:blue', label='Mean', y=mean)
# sns.lineplot(ax=g.ax_joint, x=bin_centers.flatten(), color='tab:blue', label='2*Standard deviation', y=std_up)
# sns.lineplot(ax=g.ax_joint, x=bin_centers.flatten(), color='tab:blue', y=std_down)
# sns.boxplot(data=participants, y=CLOCK, x=participants.age_bin, hue_order=age_bins, 
#                 ax=g.ax_marg_y, palette=CON_PALLETE, showfliers=False)
# sns.boxplot(data=participants, y=CLOCK, x=participants.age_bin, hue_order=age_bins, 
#                 ax=g.ax_joint, palette=CON_PALLETE, showfliers=False)
# # sns.pointplot(data=var_df, x="bin", y="variance", ax=g.ax_marg_x)
# g.refline(y=0)
# g.fig.subplots_adjust(top=0.7)

# %%  
g.savefig('../results/A_jointplot_accVSage1.svg')

# %%  Run for a clock
CLOCK = 'acc'

plt.rc('font', size=8) 

bin_means = participants.groupby('age_bin')[CLOCK].mean().values
bin_sds = participants.groupby('age_bin')[CLOCK].std().values
mean_fit = LinearRegression().fit(X = bin_centers,  y= bin_means)
sd_fit = LinearRegression().fit(X = bin_centers,  y= bin_sds)

mean = mean_fit.predict(bin_centers)
std = sd_fit.predict(bin_centers)
std_up = mean + 2*std
std_down = mean - 2*std


g = sns.JointGrid()

# joint axis
sns.scatterplot(ax=g.ax_joint, data=participants, x='age', y=CLOCK, alpha=0.3, hue='age_bin', 
                palette="rocket",  hue_order=age_bins, legend=False)
sns.lineplot(ax=g.ax_joint, x=bin_centers.flatten(), color='tab:blue', label='Mean', y=mean)
sns.lineplot(ax=g.ax_joint, x=bin_centers.flatten(), color='tab:blue', label='2*Standard deviation', y=std_up)
sns.lineplot(ax=g.ax_joint, x=bin_centers.flatten(), color='tab:blue', y=std_down)
# margins
sns.boxplot(data=participants, y=CLOCK, x=participants.age_bin, hue_order=age_bins, 
                ax=g.ax_marg_y, palette="rocket", showfliers=False)
# sns.pointplot(data=var_df, x="bin", y="variance", ax=g.ax_marg_x)
# sns.kdeplot(data=participants, y='acc', hue=participants.age_bin, hue_order=age_bins, ax=g.ax_marg_y, 
#                 palette="rocket", legend=False)
g.fig.subplots_adjust(top=0.8)
g.refline(y=0)
g.set_axis_labels(xlabel='Age (years)', 
                ylabel='Acceleration')
plt.rc('font', size=16) 

# %%  Save
g.savefig('../results/A_jointplot_accVSage_our_acc.svg')
# %%
