import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

DATA_PATH = '../exports/wave3_person_fitted.h5ad'
amdata = ad.read_h5ad(DATA_PATH)


amdata.var['age_bin'] = pd.cut(amdata.var.age, 5).astype('str')
amdata.var.age_bin= amdata.var.age_bin.astype('str')

ax = sns.kdeplot(amdata.var, x='acc', hue='age_bin')
ax.axvline(0)
plot.save(ax, 'Acc KDE age distribution', format='svg')
plot.save(ax, 'Acc KDE age distribution', format='png')

ax = sns.kdeplot(amdata.var, x='bias', hue='age_bin')
ax.axvline(0)
plot.save(ax, 'bias KDE age distribution', format='svg')
plot.save(ax, 'bias KDE age distribution', format='png')