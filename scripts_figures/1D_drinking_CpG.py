# %% ########################
# %% LOADING

# %load_ext autoreload 
# %autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

SITE_NAME = 'cg09067967'

amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/wave3_meta.h5ad', backed='r')
wave3_participants = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/wave3_participants.csv', index_col=0)
amdata = amdata[SITE_NAME].to_memory()
amdata.var.units = np.log(amdata.var.units+1)
# amdata.var.heavy_drinker.value_counts()

df = amdata[SITE_NAME].to_df().T.rename(columns={SITE_NAME:"value"})
df['units'] = np.log(1+amdata.var.units)
df['units'] = amdata.var.units
df['status']  = df.units > df.units.mean()
df['age'] = amdata.var.age
# %%
df.loc[df.units>3, 'units'] = df.units+4
# %% PLOT
g = sns.JointGrid()
g.ax_joint.set_title(SITE_NAME)
df = df.sort_values('units', ascending=True)
# df = df.sample(len(df))
sns.scatterplot(ax=g.ax_joint, data=df, x='age', y='value', hue='units',
                 palette=CON_PALLETE2, alpha=0.7, linewidth=0, legend=True)
g.ax_joint.legend(title='Units')
sns.boxplot(ax=g.ax_marg_y, data=df, y='value', x='status',
            palette=[colors[0], colors[1]],showfliers=False, legend=True)
g.fig.subplots_adjust(top=0.7)
g.ax_marg_x.remove()
g.ax_marg_y.set_frame_on(True)
g.ax_joint.set_ylabel('Methylation level (beta value)')

x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df['value'].max() , 0.01, 'k'
g.ax_marg_y.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
g.ax_marg_y.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

g.ax_joint.set_ylabel('Methylation level \n(beta value)')
g.ax_joint.set_xlabel('Age')

plot.fonts(8)
g.fig.set_size_inches((plot.cm2inch(9.5),plot.cm2inch(6)))
g.fig.tight_layout()

# %% saving
g.savefig(f'{paths.FIGURES_DIR}/fig1/1D_drinking_CpG.svg')
g.savefig(f'{paths.FIGURES_DIR}/fig1/1D_drinking_CpG.png')



# %%
from scipy.stats import ttest_ind
ttest_ind(amdata[:, df.status==True].X.flatten(),
         amdata[::, df.status==False].X.flatten())



# %%
