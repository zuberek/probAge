#%% 
# imports
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

df = pd.read_csv(f'{paths.DATA_PROCESSED_DIR}/results_for_jan.csv', index_col=0)


#%% 
# prepping
df = df[df.Clock != 'Weidner']
df['abs_smoke_beta'] = df.smoke_beta.abs()
df.replace('DeepMAge',  'DeepMAge (2021)',  inplace=True)
df.replace('Horvath',   'Horvath (2013)',   inplace=True)
df.replace('Zhang',     'Zhang (2019)',     inplace=True)
df.replace('Hannum',    'Hannum (2013)',    inplace=True)


#%% 
# plotting

g= sns.FacetGrid(df,col='Clock', col_wrap=2, height=1.5, aspect=1.5)
g.map_dataframe(sns.scatterplot, x="age_r2",  y='abs_smoke_beta', 
                alpha=0.3, size=0.2, linewidth = 0, color='tab:grey')

g.set_titles('{col_name}')
g.set_xlabels("Age association (r2)")
g.set_ylabels("Smoking \n association \n (beta)")
# g.axes[0,0].set_xlabel('axes label 1')
# g.axes[0,1].set_xlabel('axes label 2')

#%% 
# saving
g.savefig(f'{paths.FIGURES_DIR}/1B_facetgrid_clocks.svg')
g.savefig(f'{paths.FIGURES_DIR}/1B_facetgrid_clocks.png')

# %%
