# %%
import anndata as ad
import pandas as pd
from scipy.stats import f_oneway
from sklearn.feature_selection import r_regression
from sklearn.impute import KNNImputer


import numpy as np

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *


# %%
horvath = pd.read_csv('Horvath_coef.csv')

adata = ad.read_h5ad('data/tissues/age_methylation_v1.h5ad', 'r')

adata = adata[horvath.CpGmarker[1:]].to_memory()

adata = adata.T

imputer = KNNImputer(n_neighbors=2)
adata.X = imputer.fit_transform(adata.X)
adata.var['r2'] = r_regression(adata.X, adata.obs.age)**2

adata = adata[:, adata.var.r2<0.1].copy()

adata.write_h5ad('horvath_nacpgs.h5ad')

# %%

adata = ad.read_h5ad(f'{paths.DATA_PROCESSED_DIR}/tissue_horvath_filtered.h5ad')
adata.obs.tissue=adata.obs.tissue.replace('brain - cerebellum', 'cerebellum')

tissue_list = ['whole blood',
    'saliva',
    'breast',
    'cerebellum',
    'liver',
    'kidney']

tissue_data = [adata[adata.obs.tissue==tissue] for tissue in tissue_list]



statistic_list = []
pvalue_list= []

for i in range(adata.var.shape[0]):

    res = f_oneway(*[t[:,i].X.flatten() for t in tissue_data])
    statistic_list.append(res.statistic)
    pvalue_list.append(res.pvalue)

# %%
plot.fonts(8)
# ax = plot.row(figsize=(9.5*0.75, 6))
ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.despine()

sns.histplot(ax=ax, x=statistic_list, bins=20)
ax.set_xlabel('ANOVA statistic' )

ax.get_figure().tight_layout()
# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext1/Ext1A_histogram_ANOVA_tissue.png',transparent=True)
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext1/Ext1A_histogram_ANOVA_tissue.svg',transparent=True)
# %%
# Find worst site and plot

top_site = 3
i = np.argsort(np.array(statistic_list))[-top_site]

g = sns.JointGrid(ratio=2, space=0)
g.ax_joint.set_title(adata.var.index[i], y=1.3)

sns.scatterplot(ax=g.ax_joint, s=15,
                x=adata[adata.obs.tissue.isin(tissue_list)].obs.age, 
                y =adata[adata.obs.tissue.isin(tissue_list)][:, i].X.flatten(), 
                hue=adata[adata.obs.tissue.isin(tissue_list)].obs.tissue)

sns.boxplot(ax=g.ax_marg_y, y =adata[adata.obs.tissue.isin(tissue_list)][:, i].X.flatten(), 
                hue=adata[adata.obs.tissue.isin(tissue_list)].obs.tissue, showfliers=False, 
                linewidth=1, legend=False)



g.ax_joint.set_ylabel('Methylation level \n' + r'($\beta$-value)')
g.ax_joint.set_xlabel('Age (years)')

# plot.fonts(8)

g.ax_marg_x.remove()
# legend = g.ax_joint.legend(title='Tissue', loc='best', markerscale=1.5)
sns.move_legend(g.ax_joint, "lower center", title=None, 
                bbox_to_anchor=(0.5,1), ncol=3, markerscale=1.5)
g.figure.tight_layout()
# g.figure.set_size_inches((plot.cm2inch(9.5*1.25),plot.cm2inch(6)))
g.figure.set_size_inches((plot.cm2inch(9.5),plot.cm2inch(6)))

# %% saving
g.savefig(f'{paths.FIGURES_DIR}/ext1/Ext1B_tissue_plot.svg', transparent=True)
g.savefig(f'{paths.FIGURES_DIR}/ext1/Ext1B_tissue_plot.png', transparent=True)
