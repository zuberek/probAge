# %%  Imports
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

DATA_PATH = '../exports/wave3_acc.h5ad'
amdata = ad.read_h5ad(DATA_PATH)

participants = amdata.var

participants['age_bin'] = pd.cut(participants.age, 5, precision=0).astype('str')
age_bins = np.sort(participants.age_bin.unique())

from sklearn.linear_model import LinearRegression

bin_size = participants.groupby('age_bin')['pos'].count()
bin_centers = participants.groupby('age_bin')['age'].mean().values[:,np.newaxis]




# %%  Run for a clock
CLOCK = 'acc'
# CLOCK = 'Horvath'

bin_means = participants.groupby('age_bin')[CLOCK].mean().values
bin_sds = participants.groupby('age_bin')[CLOCK].std().values
mean_fit = LinearRegression().fit(X = bin_centers, y= bin_means)
sd_fit = LinearRegression().fit(X = bin_centers, y= bin_sds)

mean = mean_fit.predict(bin_centers)
std = sd_fit.predict(bin_centers)
std_up = mean + 2*std
std_down = mean - 2*std


g = sns.JointGrid()
sns.scatterplot(data=participants, x='age', y=CLOCK, alpha=0.3, hue='age_bin', 
                palette="rocket",  hue_order=age_bins, ax=g.ax_joint)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', label='Mean', y=mean, ax=g.ax_joint)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', label='2*Standard deviation', y=std_up, ax=g.ax_joint)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', y=std_down, ax=g.ax_joint)
sns.boxplot(data=participants, y=CLOCK, x=participants.age_bin, hue_order=age_bins, 
                ax=g.ax_marg_y, palette="rocket")
# sns.kdeplot(data=participants, y='acc', hue=participants.age_bin, hue_order=age_bins, ax=g.ax_marg_y, 
#                 palette="rocket", legend=False)
g.fig.suptitle(f'{CLOCK} acceleration distribution in different age bins', y=0.8)
g.fig.subplots_adjust(top=0.8)
g.refline(y=0)
