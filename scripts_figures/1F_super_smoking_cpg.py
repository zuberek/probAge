# %% ########################
# %% LOADING

# %load_ext autoreload 
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_meta.h5ad', backed='r')
wave3_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave3_participants.csv', index_col=0)

SITE_NAME= 'cg05575921'

amdata = amdata[SITE_NAME].to_memory()
amdata.var['heavy_smoker'] = amdata.var.weighted_smoke > amdata.var.weighted_smoke.std()

df = amdata[SITE_NAME].to_df().T.rename(columns={SITE_NAME:"value"})
df['weighted_smoke'] = amdata.var.weighted_smoke
df['status']  = amdata.var.weighted_smoke > amdata.var.weighted_smoke.std()
df['age'] = amdata.var.age


# %% PLOT
g = sns.JointGrid()
g.ax_joint.set_title(SITE_NAME)
# amdata = amdata[:, np.random.shuffle(amdata.var.index)].copy()
df = df.sort_values('weighted_smoke', ascending=True)
sns.scatterplot(ax=g.ax_joint, data=df, x='age', y='value', hue='weighted_smoke',
                 palette=CON_PALLETE2, alpha=1, linewidth=0, s=10, legend=True)
legend = g.ax_joint.legend(title='Smoking\n(weighted)',loc='best')
for handle in legend.legendHandles:
    handle.set_alpha(1)
    handle.set_markersize(5)
sns.boxplot(ax=g.ax_marg_y, data=df, y='value', x='status',
            palette=[colors[0], colors[1]],showfliers=False, legend=True)

g.ax_marg_x.remove()
g.ax_marg_y.set_frame_on(True)

g.ax_joint.set_ylabel('Methylation level \n' + r'($\beta$-value)')
g.ax_joint.set_xlabel('Age (years)')

# statistical annotation
x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df['value'].max() + 0.01 , 0.01, 'k'
g.ax_marg_y.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
g.ax_marg_y.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

plot.fonts(8)
g.figure.set_size_inches((plot.cm2inch(9.5),plot.cm2inch(6)))
g.figure.tight_layout()

# %% saving
g.savefig(f'{paths.FIGURES_DIR}/fig1/1F_super_smoking_cpg.svg')
g.savefig(f'{paths.FIGURES_DIR}/fig1/1F_super_smoking_cpg.png')



# %% STATISTICS

from scipy.stats import ttest_ind

ttest_ind(amdata['cg05575921', amdata.var.heavy_smoker==True].X.flatten(),
         amdata['cg05575921', amdata.var.heavy_smoker==False].X.flatten())

