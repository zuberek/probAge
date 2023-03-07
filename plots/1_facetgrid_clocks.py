#%% 
# imports
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

df = pd.read_csv('../exports/results_for_jan.csv', index_col=0)
df = df[df.Clock != 'Weidner']
df['abs_smoke_beta'] = df.smoke_beta.abs()
#%% 
# plotting

g= sns.FacetGrid(df,col='Clock', col_wrap=2, height=1.5, aspect=1.5)
g.map_dataframe(sns.scatterplot, x="age_r2", size=0.2, linewidth = 0, y='abs_smoke_beta', color='tab:grey')

g.savefig('../results/1_facetgrid_clocks.svg')
g.savefig('../results/1_facetgrid_clocks.png')
