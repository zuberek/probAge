# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

amdata = ad.read_h5ad('../exports/wave3_meta.h5ad')

amdata = amdata[:, amdata.var.sort_values('weighted_smoke', ascending=True).index]
#%%
# Plot Zhang site


site_index = 'cg24090911'
ax=plot.row('')
sns.scatterplot(x=amdata.var.age, y=amdata[site_index].X.flatten(), 
                hue=amdata.var.weighted_smoke.values, palette="RdBu")

norm = plt.Normalize(amdata.var.weighted_smoke.min(), amdata.var.weighted_smoke.max())
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm)


#%%
site_indexes =['cg05575921','cg09067967','cg24090911']