#%% imports
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling_bio
import arviz as az

from sklearn.linear_model import LinearRegression

amdata = ad.read_h5ad('../exports/wave3_meta.h5ad', backed='r')
# amdata_bio = ad.read_h5ad('../exports/wave3_acc.h5ad')

horvath_coef = pd.read_csv('../resources/Horvath_coef.csv', index_col=0)
intersection = amdata.obs.index.intersection(horvath_coef.index)
amdata_horvath = amdata[intersection].to_memory()

horvath_data = pd.read_csv('../exports/GS-P2-Wave3-Phe_hyperHypo.csv', index_col=0)
horvath_data.columns[150:200]
age = horvath_data['age'].values
horvath_data = horvath_data.iloc[:,3:]
ages = np.broadcast_to(age, (horvath_data.shape[1], horvath_data.shape[0])).T
horvath_data = horvath_data - ages

# %%
clocks = ['Horvath', 'Hannum']
# clocks = ['Hannum','SkinClock','Horvath']
offsets=[0.01,0.02,0.03,0.04,0.05,0.15]
offsets=[0.02,0.05]
all_offsets = sorted([-offset for offset in offsets]) + [0] + offsets

all_clock_data = pd.DataFrame()
for clock in clocks:
    columns =   [f'{int(offset*100)}%MinusAbs-{clock}' for offset in sorted(offsets, reverse=True)] + \
            [clock] + \
            [f'{int(offset*100)}%PlusAbs-{clock}' for offset in offsets] 
    clock_data = horvath_data[columns].copy()
    clock_data.columns = all_offsets
    clock_data['clock'] = clock
    all_clock_data = pd.concat([all_clock_data, clock_data])

# all_clock_data['shift'] = all_clock_data.groupby('clock')[0].transform('mean')
# all_clock_data[all_offsets] = all_clock_data[all_offsets].values - np.broadcast_to(all_clock_data['shift'], 
#                                                        ( len(all_offsets), all_clock_data.shape[0])).T

shifts = all_clock_data.groupby('clock')[0].transform('mean')
all_clock_data[all_offsets] = all_clock_data[all_offsets].values - np.broadcast_to(shifts, 
                                                       ( len(all_offsets), all_clock_data.shape[0])).T

df = all_clock_data.melt(id_vars='clock', var_name='offset', value_name='age acceleration')
# sns.boxplot(data=df, x='clock', y='age acceleration', hue='offset',  showfliers=False)
# %%
ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.boxplot(data=df, x='offset', y='age acceleration', 
            hue='clock', showfliers=False, ax=ax)
# sns.catplot(kind='box', data=df, x='offset', y='age acceleration', 
#             hue='clock', width=0.6, showfliers=False, ax=ax)
ax.axhline(y=0, dashes=[5,3], color='Grey')
sns.despine()

ax.legend(title=None, loc='best')

ax.set_ylabel('Age acceleration (years)')
ax.set_xlabel('Applied global methylation bias\n' + r'($\beta$-value)')
ax.get_figure().tight_layout()

# %% saving
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig1/1G_global_bias_problem.svg')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/fig1/1G_global_bias_problem.png')

