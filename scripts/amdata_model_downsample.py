%load_ext autoreload
%autoreload 2
sys.path.append("..")   # fix to import modules from root
from turtle import color
from src.general_imports import *

from src import modelling

# import logging
# logger = logging.getLogger('pymc')
# logger.propagate = False
# logger.setLevel(logging.ERROR)

amdata = amdata.AnnMethylData('../exports/wave3_linear.h5ad')



# maps = modelling.vectorize_all_participants(amdata)
# amdata.participants['acc'] = maps['acc']
# amdata.participants['bias'] = maps['bias']

# sns.jointplot(data=amdata.participants, x='acc', y='bias')
# ax1, ax2 = plot.row([])


n_sites_grid = [2**n for n in range(1,11)]
n_sites_label = [f'{n_sites_grid[i]}-{n_sites_grid[i+1]}' for i in range(len(n_sites_grid))[:-1]]

accs = np.empty((len(n_sites_grid), amdata.n_participants))
biases = np.empty((len(n_sites_grid), amdata.n_participants))
for i, n_sites in enumerate(tqdm(n_sites_grid)):
    map = modelling.vector_person_model(amdata[:n_sites], return_MAP=True, return_trace=False)['map']
    accs[i] = map['acc']
    biases[i] = map['bias']    


accs = pd.DataFrame(accs, index=n_sites_grid, columns=amdata.participants.index).T
biases = pd.DataFrame(biases, index=n_sites_grid, columns=amdata.participants.index).T
accs.to_csv('../exports/downsample_accs.csv')
biases.to_csv('../exports/downsample_biases.csv')
accs = pd.read_csv('../exports/downsample_accs.csv', index_col=0)
biases = pd.read_csv('../exports/downsample_biases.csv', index_col=0)

r2_array = pd.Series([amdata.sites.iloc[i].r2 for i in n_sites_grid[:-1]], name='r2 value')

ax1, ax2 = plot.row(['Acceleration', 'Bias'], scale=10)
df = pd.DataFrame(np.abs(np.diff(accs, axis=1)), index=amdata.participants.index, columns=n_sites_label)
# df_melted = df.reset_index().melt(id_vars='index').rename(columns={'variable':'Site Interval', 'value': 'Mean absolute difference in interval'})
ax11 = ax1.twinx()
sns.boxplot(df, color='tab:blue', ax=ax1)
# sns.pointplot(data=df_melted, x='Site Interval', y='Mean absolute difference in interval', ax=ax1)
# sns.pointplot(data=df_melted, x='Site Interval', y='Mean absolute difference in interval', ax=ax1)
sns.lineplot(x=n_sites_label, y=r2_array, marker='o', color='tab:orange', label='r2 value', ax=ax11)

pd.DataFrame(biases, index=n_sites_grid, columns=amdata.participants.index)
df = pd.DataFrame(np.abs(np.diff(biases, axis=1)), index=amdata.participants.index, columns=n_sites_label)
df_melted = df.reset_index().melt(id_vars='index').rename(columns={'variable':'Site Interval', 'value': 'Mean absolute difference in interval'})
ax22 = ax2.twinx()
sns.boxplot(df, color='tab:blue',  ax=ax2)
# sns.pointplot(data=df_melted, x='Site Interval', y='Mean absolute difference in interval', ax=ax2)
sns.lineplot(x=n_sites_label, y=r2_array, marker='o', color='tab:orange', label='r2 value', ax=ax22)


df.values.argmax(axis=0)
df.iloc[3258]
accs.iloc[3148]