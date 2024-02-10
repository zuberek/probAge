# %%
# requires running the person_model.py main script

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

participants = pd.read_csv('../exports/wave3_participants.csv', index_col=0)

N_BINS = 5
participants['age_bin'] = pd.qcut(participants.age, N_BINS)
participants['age_bin'] = participants['age_bin'].apply(
    lambda x: f'({int(round(x.left))}, {int(round(x.right))}]')
participants = participants.sort_values('age')
participants['acc_center'] = participants.acc - participants.groupby('age_bin')['acc'].transform('mean')

participants['Horvath_scaled'] = participants.Horvath/participants.Horvath.std()
participants['acc_scaled'] = participants.acc/participants.acc.std()

# %%
# Boxplot showing our acceleration variance change next to Horvath
ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.despine()

# shift to be around zero
participants.acc_scaled = participants.acc_scaled - participants.groupby('age_bin')['acc_scaled'].transform('mean')
participants.Horvath_scaled = participants.Horvath_scaled - participants.groupby('age_bin')['Horvath_scaled'].transform('mean')

df = participants.reset_index().melt(id_vars='age_bin', value_vars=['acc_scaled', 'Horvath_scaled'])
sns.boxplot(data=df, x='age_bin', y='value', hue='variable', showfliers=False)
sns.despine()
# %%
# Regression line to data
from sklearn.linear_model import LinearRegression

acc_bin_means = participants.groupby('age_bin')['acc'].mean()
Horvath_bin_means = participants.groupby('age_bin')['Horvath'].mean()

bin_size = participants.groupby('age_bin')['name'].count()
bin_centers = participants.groupby('age_bin')['age'].mean().values[:,np.newaxis]

acc_bin_means = participants.groupby('age_bin')['acc_scaled'].mean().values
acc_bin_sds = participants.groupby('age_bin')['acc_scaled'].std().values
acc_mean_fit = LinearRegression().fit(X = bin_centers,  y= acc_bin_means)
acc_sd_fit = LinearRegression().fit(X = bin_centers,  y= acc_bin_sds)

acc_mean = acc_mean_fit.predict(bin_centers)
acc_std = acc_sd_fit.predict(bin_centers)
acc_std_up = acc_mean + 2*acc_std
acc_std_down = acc_mean - 2*acc_std

sns.lineplot(x=bin_centers.flatten(), color='tab:blue', label='Mean', y=acc_mean)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', label='2*Standard deviation', y=acc_std_up)
sns.lineplot(x=bin_centers.flatten(), color='tab:blue', y=acc_std_down)

Horvath_bin_means = participants.groupby('age_bin')['Horvath_scaled'].mean().values
Horvath_bin_sds = participants.groupby('age_bin')['Horvath_scaled'].std().values
Horvath_mean_fit = LinearRegression().fit(X = bin_centers,  y= Horvath_bin_means)
Horvath_sd_fit = LinearRegression().fit(X = bin_centers,  y= Horvath_bin_sds)

Horvath_mean = Horvath_mean_fit.predict(bin_centers)
Horvath_std = Horvath_sd_fit.predict(bin_centers)
Horvath_std_up = Horvath_mean + 2*Horvath_std
Horvath_std_down = Horvath_mean - 2*Horvath_std

sns.lineplot(x=bin_centers.flatten(), color='tab:red', label='Mean', y=Horvath_mean)
sns.lineplot(x=bin_centers.flatten(), color='tab:red', label='2*Standard deviation', y=Horvath_std_up)
sns.lineplot(x=bin_centers.flatten(), color='tab:red', y=Horvath_std_down)

# %%
# Regression line to variance
acc_bin_vars = participants.groupby('age_bin')['acc_scaled'].var().values
acc_mean_fit = LinearRegression().fit(X = bin_centers,  y= acc_bin_means)

acc_mean = acc_mean_fit.predict(bin_centers)


sns.lineplot(x=bin_centers.flatten(), color='tab:blue', label='mean_acc', y=acc_mean)

Horvath_bin_vars = participants.groupby('age_bin')['Horvath_scaled'].mean().values
Horvath_mean_fit = LinearRegression().fit(X = bin_centers,  y= Horvath_bin_means)

Horvath_mean = Horvath_mean_fit.predict(bin_centers)

sns.lineplot(x=bin_centers.flatten(), color='tab:red', label='mean_horvath', y=Horvath_mean)
# %%
ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.despine()
ax.get_figure().tight_layout()
ax.set_xlabel('Age bin')
ax.set_ylabel('Acceleration')

sns.boxplot(data=participants, x='age_bin', y='acc', showfliers=False)

# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/Ext3F_boxplot_accvarVSage.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/Ext3F_boxplot_accvarVSage.svg')
