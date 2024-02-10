#%% imports
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling
import arviz as az

from sklearn.linear_model import LinearRegression

amdata = ad.read_h5ad('../exports/wave3_person_fitted.h5ad')
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

horvath_data['Horvath'].hist()

# %%
clocks = ['Horvath', 'Hannum', 'acc']
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
ax=plot.row('')
ax.axhline(y=0, dashes=[5,3], color='tab:grey')
sns.boxplot(data=df, x='offset', y='age acceleration', hue='clock',  showfliers=False, ax=ax)
sns.despine()

# %% saving
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/1G_global_bias_problem.svg')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/1G_global_bias_problem.png')

# plot.save(ax, '1_clock_global_offset_problem', format=['png','svg'])
# %%

accs =[]

for offset in tqdm(all_offsets):
    amdata_offset = amdata.copy()
    amdata_offset.X = amdata.X+offset
    amdata_offset.X = np.where(amdata_offset.X <= 0, 0.00001, amdata_offset.X)
    amdata_offset.X = np.where(amdata_offset.X >= 1, 0.99999, amdata_offset.X)
    map = modelling.person_model(amdata_offset,  method="map", map_method='L-BFGS-B')
    accs.append(map['acc'])


accs = pd.DataFrame(accs, index=np.round(all_offsets,2)).T
accs.mean(axis=0)
shifts = np.broadcast_to(accs.mean(axis=0), (accs.shape[0], len(all_offsets)))
shifts.shape
pd.DataFrame(accs - shifts, columns=np.round(all_offsets,2)).T[0].hist(bins=50)
pd.DataFrame(accs  , index=np.round(all_offsets,2)).T.hist(bins=50)
(accs-shifts).hist(bins=50)
accs.mean()
accss=accs
accs=accss
accs = accs-shifts


accs_df=accs.melt(var_name='offset', value_name='acc').assign(type='acc')
accs_df['std'] = accs_df.groupby('offset')['acc'].transform('std')
accs_df['normalised_acc'] = accs_df.acc/accs_df['std']

# %%
horvaths_df = all_clock_data[all_clock_data.clock=='Horvath'].copy()
horvaths_df=horvaths_df[all_offsets].melt(var_name='offset', value_name='acc').assign(type='horvath')
horvaths_df['std'] = horvaths_df.groupby('offset')['acc'].transform('std')
horvaths_df['normalised_acc'] = horvaths_df.acc/horvaths_df['std']

Hannums_df = all_clock_data[all_clock_data.clock=='Hannum'].copy()
Hannums_df=Hannums_df[all_offsets].melt(var_name='offset', value_name='acc').assign(type='Hannum')
Hannums_df['std'] = Hannums_df.groupby('offset')['acc'].transform('std')
Hannums_df['normalised_acc'] = Hannums_df.acc/Hannums_df['std']

Hannums_df.normalised_acc = Hannums_df.normalised_acc - Hannums_df.normalised_acc.mean()
horvaths_df.normalised_acc = horvaths_df.normalised_acc - horvaths_df.normalised_acc.mean()
accs_df.normalised_acc = accs_df.normalised_acc - accs_df.normalised_acc.mean()

df = pd.concat((accs_df, horvaths_df, Hannums_df), axis=0)
df = df[['type','offset','normalised_acc']]
df.loc[df.type=='acc', 'type'] = 'Acceleration'
df.loc[df.type=='horvath', 'type'] = 'Horvath (2013)'
df.loc[df.type=='Hannum', 'type'] = 'Hannum (2013)'

df.to_csv('../exports/wave3_acc_global_offsets_problem.csv')

# %% plot

ax = plot.row(figsize=(9.5, 6))
plot.fonts(8)
sns.despine()

ax=sns.boxplot(data=df, x='offset', hue_order=['Acceleration', 'Horvath (2013)', 'Hannum (2013)'],
                y='normalised_acc', hue='type', showfliers=False)
ax.axhline(y=0, dashes=[5,3], color='tab:grey')

ax.set_xlabel('Offset')
ax.set_ylabel('Normalised acceleration')
ax.legend(title='Clock')

plt.tight_layout()

# %% save
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/Ext3D_global_bias_problem.png')
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/Ext3D_global_bias_problem.svg')





















# #%% compute accelerations
# global_offsets = np.array([-0.1, -0.05, 0, 0.05, 0.1])

# mod=LinearRegression().fit(X=amdata_horvath.X.T, y=amdata_horvath.var.age)
# amdata_horvath.var['new_Horvath']=mod.predict(amdata_horvath.X.T)-amdata.var.age
# sns.scatterplot(data=amdata_horvath.var, x='Horvath', y='new_Horvath')

# accs =[]
# horvaths = []

# for offset in tqdm(global_offsets):
#     amdata_bio_offset = amdata_bio.copy()
#     amdata_horvath_offset = amdata_horvath.copy()
#     amdata_bio_offset.X = amdata_bio.X+offset
#     amdata_horvath_offset.X = amdata_horvath.X+offset
#     maps = modelling_bio.person_model(amdata_bio_offset, return_trace=False, return_MAP=True, show_progress=False, cores=4)['map']
#     accs.append(maps['acc'])
#     horvaths.append((mod.predict(amdata_horvath_offset.X.T)-amdata_horvath.var.age).values)


# #%% transform output
# accs_df=pd.DataFrame(accs, index=np.round(global_offsets,2)).reset_index().melt(id_vars=['index'], var_name='offset').assign(type='acc')
# horvaths_df=pd.DataFrame(horvaths, index=np.round(global_offsets,2)).reset_index().melt(id_vars=['index'], var_name='offset').assign(type='horvath')

# accs_df['std'] = accs_df.groupby('index')['value'].transform('std')
# accs_df['normalised_value'] = accs_df.value/accs_df['std']
# horvaths_df['std'] = horvaths_df.groupby('index')['value'].transform('std')
# horvaths_df['normalised_value'] = horvaths_df.value/horvaths_df['std']

# df = pd.concat((accs_df, horvaths_df), axis=0)
# df = df.drop(['offset'], axis=1).rename(columns={'index': 'offset'})

# #%% plot
# ax=sns.boxplot(data=df, x='offset', 
#                 y='normalised_value', hue='type', showfliers=False)

# #%% save
# plot.save(ax, 'A_Horvath_bias_problem', format='svg')
# plot.save(ax, 'A_Horvath_bias_problem', format='png')
# %%