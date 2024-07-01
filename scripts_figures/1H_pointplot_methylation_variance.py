# %%

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import src.modelling_bio_beta as modelling

amdata = ad.read_h5ad('../exports/wave3_meta.h5ad', backed='r')
# part_indexes = modelling.sample_to_uniform_age(amdata, n_part=2000)
# amdata[:, part_indexes].var.age.hist()
# amdata = amdata[:, part_indexes]

# Take only the sites correlated with age
amdata = amdata[amdata.obs.r2>0.1].to_memory()

# Cut off age to make the bins have more participants
amdata = amdata[:, amdata.var.age<80].copy()

N_BINS = 5
amdata.var['age_bin'] = pd.cut(amdata.var.age, N_BINS)
# amdata.var['age_bin'] = pd.qcut(amdata.var.age, N_BINS)

# round age bins to int
amdata.var['age_bin'] = amdata.var['age_bin'].apply(
    lambda x: f'({round(x.left,1)}, {round(x.right,1)}]')
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

# %%
var_df = pd.DataFrame()
for age_bin in age_bins:
    bin = pd.DataFrame(amdata[:,amdata.var.age_bin == age_bin].X.var(axis=1).tolist(), columns=['variance'])
    bin['bin'] = age_bin
    var_df= pd.concat((var_df, bin), axis=0)

var_df['sd'] = np.sqrt(var_df.variance)
# %%
plot.fonts(8)
ax = plot.row(figsize=(9.5, 6))
sns.despine()
ax.set_xlabel('Age bin (years)' )
ax.set_ylabel('Methylation variance \n' + r'($\beta$-value)' )

sns.pointplot(ax=ax, data=var_df, x="bin", y="variance", 
                 errorbar='ci', capsize=.3, linewidth=1)
# ax.ticklabel_format(style='sci')
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax.get_figure().tight_layout()
# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig1/1H_pointplot_methylation_variance.png', transparent=True)
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig1/1H_pointplot_methylation_variance.svg', transparent=True)

# %%
# Boxplot, not used
# ax = plot.row(figsize=(3.6, 2.6))
# sns.boxplot(ax=ax, data=var_df, x="bin", y="sd", showfliers=False, linewidth=1, showmeans=True)
# sns.lineplot(ax=ax, data=var_df.groupby('bin').mean(), x='bin', y='sd')
# ax.set_xlabel('Age bin')
# sns.despine()




