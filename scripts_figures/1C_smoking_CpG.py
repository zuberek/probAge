# %% LOADING

# %load_ext autoreload 
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_meta.h5ad', backed='r')
wave3_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave3_participants.csv', index_col=0)
SITE_NAME = 'cg24090911'

amdata = amdata[SITE_NAME].to_memory()
amdata.var['heavy_smoker'] = amdata.var.weighted_smoke > amdata.var.weighted_smoke.std()

df = amdata[SITE_NAME].to_df().T.rename(columns={"cg24090911":"value"})
df['weighted_smoke'] = amdata.var.weighted_smoke
df['status']  = amdata.var.weighted_smoke > amdata.var.weighted_smoke.quantile(0.85)
df['age'] = amdata.var.age
# %% PLOT
g = sns.JointGrid()


g.ax_joint.set_title(SITE_NAME)
# amdata = amdata[:, np.random.shuffle(amdata.var.index)].copy()
df = df.sort_values('weighted_smoke', ascending=True)
sns.scatterplot(ax=g.ax_joint, data=df, x='age', y='value', hue='weighted_smoke',
                 palette=CON_PALLETE2, alpha=1, linewidth=0, legend=True, s=10)
legend = g.ax_joint.legend(title='Smoking\n(weighted)',loc='best')

for handle in legend.legendHandles:
    handle.set_alpha(1)
    handle.set_markersize(5)
sns.boxplot(ax=g.ax_marg_y, data=df, y='value', x='status',
            palette=[colors[0], colors[1]],showfliers=False, legend=True)
# g.figure.subplots_adjust(top=0.7)
g.ax_marg_x.remove()
g.ax_marg_y.set_frame_on(True)
# g.ax_joint.spines[['right']].set_visible(True)

g.ax_joint.set_ylabel('Methylation level \n' + r'($\beta$-value)')
g.ax_joint.set_xlabel('Age (years)')
# sns.despine(g.fig)
# g.ax_marg_y.legend(title='Smoker', loc='lower right', labels=['Heavy', 'None'])
# statistical annotation


x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df['value'].max() , 0.01, 'k'
g.ax_marg_y.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
g.ax_marg_y.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

plot.fonts(8)
g.figure.set_size_inches((plot.cm2inch(9.5),plot.cm2inch(6)))
g.figure.tight_layout()

# %% saving
g.savefig(f'{paths.FIGURES_DIR}/fig1/1C_smoking_CpG.svg', transparent=True)
g.savefig(f'{paths.FIGURES_DIR}/fig1/1C_smoking_CpG.png', transparent=True)


# %% STATISTICS

from scipy.stats import ttest_ind

ttest_ind(amdata['cg24090911', amdata.var.heavy_smoker==True].X.flatten(),
         amdata['cg24090911', amdata.var.heavy_smoker==False].X.flatten())

# %%
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale

df = amdata.to_df().T
df[['age', 'heavy_smoker', 'weighted_smoke']] = amdata.var[['age', 'heavy_smoker', 'weighted_smoke']]
df['age_scaled'] = df.age / df.age.max()
smf.ols("cg24090911 ~   scale(weighted_smoke) + scale(age)", df).fit().summary()

# %%


from scipy.stats import linregress

values = np.argwhere(~np.isnan(amdata.var.weighted_smoke)).flatten()

res = linregress(amdata.X.flatten()[values], amdata.var.weighted_smoke.iloc[values])
res.rvalue**2

# %%
# correlation with age
res = linregress(amdata.X.flatten(), amdata.var.age)
res.rvalue**2