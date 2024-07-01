# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import src.modelling_bio_beta as modelling

amdata = ad.read_h5ad('../exports/wave3_person_fitted.h5ad', backed='r')
# part_indexes = modelling.sample_to_uniform_age(amdata, n_part=2000)
# amdata[:, part_indexes].var.age.hist()
# amdata = amdata[:, part_indexes]


# Cut off age to make the bins have more participants
amdata = amdata[:, amdata.var.age<80].to_memory()

N_BINS = 5
amdata.var['age_bin'] = pd.cut(amdata.var.age, N_BINS)
# amdata.var['age_bin'] = pd.qcut(amdata.var.age, N_BINS)

# round age bins to int
amdata.var['age_bin'] = amdata.var['age_bin'].apply(
    lambda x: f'({round(x.left)}, {round(x.right)}]')
age_bins = np.sort(amdata.var.age_bin.unique())

max_part = amdata.var.age_bin.value_counts().min()

# Run only for the sites in Horvath
# site_meta = pd.read_csv('../resources/AllGS_CpGs.csv', index_col=0)
# site_hor = site_meta[site_meta.Clock=='Horvath'].index
# intersection = amdata.obs.index.intersection(site_hor)
# amdata = amdata[intersection].to_memory()


part_index = []
for bin in amdata.var.age_bin.unique():
    age_bin = amdata[:, amdata.var.age_bin == bin].copy()
    part_index.extend(list(np.random.choice(age_bin.var.index, max_part, replace=False)))

amdata = amdata[:, part_index].copy()

participants = amdata.var
participants['Horvath'] = participants.Horvath/participants.Horvath.std()
participants['Acceleration'] = participants.acc/participants.acc.std()

# %%
# Boxplot showing our Accelerationeleration variance change next to Horvath
ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.despine()

# shift to be around zero
participants.Acceleration = participants.Acceleration - participants.groupby('age_bin')['Acceleration'].transform('mean')
participants.Horvath = participants.Horvath - participants.groupby('age_bin')['Horvath'].transform('mean')

df = participants.reset_index().melt(id_vars='age_bin', value_vars=['Acceleration', 'Horvath'])
sns.boxplot(data=df, x='age_bin', y='value', hue='variable', showfliers=False)
sns.despine()

ax.set_xlabel('Age bin (years)')
ax.set_ylabel('Acceleration')
ax.get_figure().tight_layout()
legend = ax.legend(title='Clock', bbox_to_anchor=(1,1), loc='best')
ax.tick_params(axis='x', labelrotation=30)
ax.get_figure().tight_layout()
# ax2.set_ylabel('Methylation level \n(beta values)')

# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3D_boxplot_accvarVSage.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext3/Ext3D_boxplot_accvarVSage.svg')

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
