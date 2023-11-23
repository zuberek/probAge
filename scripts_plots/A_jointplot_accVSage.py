# %%  
# Imports
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

DATA_PATH = '../exports/wave3_acc.h5ad'
amdata = ad.read_h5ad(DATA_PATH)

N_BINS = 5

participants = amdata.var

participants['age_bin'] = pd.cut(participants.age, N_BINS, precision=0).astype('str')
age_bins = np.sort(participants.age_bin.unique())

from sklearn.linear_model import LinearRegression

bin_size = participants.groupby('age_bin')['pos'].count()
bin_centers = participants.groupby('age_bin')['age'].mean().values[:,np.newaxis]

var_df = pd.DataFrame()
for age_bin in age_bins:
    bin = pd.DataFrame(amdata[:,amdata.var.age_bin == age_bin].X.var(axis=1).toarray(), columns=['variance'])
    bin['bin'] = age_bin
    var_df= pd.concat((var_df, bin), axis=0)


# %%  
# Run for a our acceleration
CLOCK = 'acc'

bin_means = participants.groupby('age_bin')[CLOCK].mean().values
bin_sds = participants.groupby('age_bin')[CLOCK].std().values
mean_fit = LinearRegression().fit(X = bin_centers,  y= bin_means)
sd_fit = LinearRegression().fit(X = bin_centers,  y= bin_sds)

mean = mean_fit.predict(bin_centers)
std = sd_fit.predict(bin_centers)
std_up = mean + 2*std
std_down = mean - 2*std


g = sns.JointGrid(ylim=(-20,20))
sns.scatterplot(data=participants, x='age', y=CLOCK, alpha=0.3, hue='age_bin', 
                palette=CON_PALLETE,  hue_order=age_bins, ax=g.ax_joint, legend=False)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', label='Mean', y=mean, ax=g.ax_joint)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', label='2*Standard deviation', y=std_up, ax=g.ax_joint)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', y=std_down, ax=g.ax_joint)
sns.boxplot(data=participants, y=CLOCK, x=participants.age_bin, hue_order=age_bins, 
                ax=g.ax_marg_y, palette=CON_PALLETE, showfliers=False)
sns.boxplot(data=participants, y=CLOCK, x=participants.age_bin, hue_order=age_bins, 
                ax=g.ax_joint, palette=CON_PALLETE, showfliers=False)
sns.pointplot(data=var_df, x="bin", y="variance", ax=g.ax_marg_x)
g.refline(y=0)
g.fig.subplots_adjust(top=0.7)

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


g = sns.JointGrid(fontsize=8)
sns.scatterplot(data=participants, x='age', y=CLOCK, alpha=0.3, hue='age_bin', 
                palette="rocket",  hue_order=age_bins, ax=g.ax_joint, legend=False)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', label='Mean', y=mean, ax=g.ax_joint)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', label='2*Standard deviation', y=std_up, ax=g.ax_joint)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', y=std_down, ax=g.ax_joint)
sns.boxplot(data=participants, y=CLOCK, x=participants.age_bin, hue_order=age_bins, 
                ax=g.ax_marg_y, palette="rocket", showfliers=False)
# sns.kdeplot(data=participants, y='acc', hue=participants.age_bin, hue_order=age_bins, ax=g.ax_marg_y, 
#                 palette="rocket", legend=False)
g.fig.subplots_adjust(top=0.8)
g.refline(y=0)
g.set_axis_labels(xlabel='Age (years)', 
                ylabel='Acceleration')
plt.rc('font', size=16) 
g.savefig('../results/A_jointplot_accVSage_our_acc.svg')
# %%
