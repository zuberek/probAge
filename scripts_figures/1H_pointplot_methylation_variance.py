# %%

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

amdata = ad.read_h5ad('../exports/wave3_meta.h5ad', backed='r')

N_BINS = 5
# amdata.var['age_bin'] = pd.cut(amdata.var.age, N_BINS).astype('str')
amdata.var['age_bin'] = pd.qcut(amdata.var.age, N_BINS)
amdata.var['age_bin'] = amdata.var['age_bin'].apply(
    lambda x: f'({int(round(x.left))}, {int(round(x.right))}]')
age_bins = np.sort(amdata.var.age_bin.unique())
int(round(2.5))
site_meta = pd.read_csv('../resources/AllGS_CpGs.csv', index_col=0)
site_hor = site_meta[site_meta.Clock=='Horvath'].index
intersection = amdata.obs.index.intersection(site_hor)
amdata = amdata[intersection].to_memory()

# %%
var_df = pd.DataFrame()
for age_bin in age_bins:
    bin = pd.DataFrame(amdata[:,amdata.var.age_bin == age_bin].X.var(axis=1).tolist(), columns=['variance'])
    bin['bin'] = age_bin
    var_df= pd.concat((var_df, bin), axis=0)

var_df['sd'] = np.sqrt(var_df.variance)
# %%
ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.despine()
ax.set_xlabel('Age bin')
ax.set_ylabel('Methylation variance in bin')

sns.pointplot(ax=ax, data=var_df, x="bin", y="variance", 
                 errorbar='ci', capsize=.3, linewidth=1)

ax.get_figure().tight_layout()
# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig1/1H_pointplot_methylation_variance.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig1/1H_pointplot_methylation_variance.svg')

# %%
# Boxplot, not used
# ax = plot.row(figsize=(3.6, 2.6))
# sns.boxplot(ax=ax, data=var_df, x="bin", y="sd", showfliers=False, linewidth=1, showmeans=True)
# sns.lineplot(ax=ax, data=var_df.groupby('bin').mean(), x='bin', y='sd')
# ax.set_xlabel('Age bin')
# sns.despine()




