#%% 
# Imports
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import preprocess_func
from sklearn.feature_selection import r_regression
from src import modelling_bio

#%% 
# Preprocess (one time run)

df = pd.read_csv('../data/DS_Blood_Fullarray.csv', index_col=0)
meta = pd.read_csv('../data/GSE52588_DownSyndrome_hyperhypo.csv', index_col=0)

# numbers = meta.title.str.split(',', expand=True)[2].str.split(' ', expand=True)[1].tolist()

# cols = [f'{num} Methylated Signal' for num in numbers]
# df = df[cols]
# df.columns = meta.index
# meta['characteristics_ch1.3'].str.split(': ', expand=True)[1]
# meta

amdata = ad.AnnData(df)
amdata.var['sex'] = meta['characteristics_ch1.3'].str.split(': ', expand=True)[1]
amdata.var['age'] = meta['characteristics_ch1.4'].str.split(': ', expand=True)[1].astype('int')
amdata.var['status'] = meta['characteristics_ch1.1'].str.split(': ', expand=True)[1]
amdata.var['family'] = meta['characteristics_ch1.2'].str.split(': ', expand=True)[1]

amdata = preprocess_func.drop_nans(amdata)
amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)

r2 = []
for site_index in tqdm(amdata.obs.index):
    r2.append(r_regression(amdata[site_index].X.T, amdata.var.age)[0]**2)
amdata.obs['r2_retrained'] = r2

clocks = ['Horvath', 'SkinClock', 'DNAmPhenoAge', 'Weidnerpred3', 'hannum_pred_age']
amdata.var[clocks] = meta[clocks]

amdata.write_h5ad('../exports/downsyndrome_full.h5ad')

#%% 
# Load

amdata_path = '../exports/downsyndrome_full.h5ad'
site_info_path = '../exports/wave3_acc_sites.csv' 

# Load intersection of sites in new dataset
wave3 = ad.read_h5ad('../exports/wave3_MAP_acc.h5ad', backed='r')

site_info = pd.read_csv(site_info_path, index_col=0)
amdata = ad.read_h5ad(amdata_path, 'r')
params = modelling_bio.get_site_params()

intersection = amdata.obs.index.intersection(site_info.index)
amdata = amdata[intersection].to_memory()

# Add ground truth from genscot
amdata.obs[params + ['r2']] = site_info[params + ['r2']]

# Infer the offsets
healthy = amdata[:, amdata.var.status=='healthy'].copy()
maps = modelling_bio.site_offsets(healthy, return_MAP=True, return_trace=False, show_progress=True)['map']
amdata.obs['offset'] = maps['offset']

amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset

sns.histplot(amdata.obs.offset)
amdata = amdata[amdata.obs.sort_values('offset', ascending=True).index]

#%% 
# Plot
ax = plot.row('')
site_index = amdata.obs.index[0]
sns.scatterplot(x=wave3.var.age,  y=wave3[site_index].X.flatten(), label='Reference',ax=ax)
sns.scatterplot(x=amdata.var.age,  y=amdata[site_index].X.flatten(), label='Not corrected', ax=ax)
sns.scatterplot(x=amdata.var.age,  y=amdata[site_index].X.flatten(), hue=amdata.var.status)

#%%
# Calculate the acc and bias
ab_maps = modelling_bio.person_model(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']

amdata.var['acc'] = ab_maps['acc']
amdata.var['bias'] = ab_maps['bias']

df = pd.concat(
    [wave3.var[['acc','bias']].assign(status='Reference'),
    amdata.var[['acc','bias','status']]]
)

#%%
g = sns.JointGrid()
sns.scatterplot(data=amdata.var, x='acc', y='bias', hue='status', ax=g.ax_joint)
# sns.kdeplot(data=amdata.var, x='acc', y='bias', hue='status', ax=g.ax_joint)
sns.kdeplot(data=amdata.var,  x='acc', common_norm=True, legend=False, hue='status',ax=g.ax_marg_x)
sns.kdeplot(data= amdata.var, y='bias', common_norm=True, legend=False, hue='status',ax=g.ax_marg_y)
g.ax_marg_x.vlines(amdata.var.groupby('status')['acc'].mean(), ymin=0, ymax=1, colors=colors)
g.ax_marg_y.hlines(amdata.var.groupby('status')['bias'].mean(), xmin=0, xmax=20, colors=colors)

g.fig.set_figwidth(3.6)
g.fig.set_figheight(2.6)
g.set_axis_labels(xlabel='Acceleration', ylabel='Bias')
g.savefig('../results/downsyndrome.svg')
#%%
sns.scatterplot(amdata.var, x='acc', y='bias', hue='status', size='age')
amdata.var.groupby('status')['acc'].mean()

#%%
from scipy.stats import f_oneway

acc_control = amdata[:, amdata.var.status == 'healthy'].var['acc'].values
acc_down = amdata[:, amdata.var.status == 'Down syndrome'].var['acc'].values


acc_control = amdata[:, amdata.var.status == 'healthy'].var['hannum_pred_age'].values
acc_down = amdata[:, amdata.var.status == 'Down syndrome'].var['hannum_pred_age'].values
f_oneway(acc_control, acc_down)

from statsmodels.formula.api import logit
amdata.var['status_binary'] = 1

amdata.var.loc[amdata.var.status=='healthy', 'status_binary'] = 0
females = amdata.var[amdata.var.sex=='Female']
model = logit('status_binary ~ scale(Horvath) + sex + age', data=amdata.var).fit()
model = logit('status_binary ~ scale(acc)', data=amdata.var).fit()
model = logit('status_binary ~ scale(acc)+ scale(bias)  ', data=amdata.var).fit()
model.summary()

amdata.var['Horvath_acc'] = amdata.var.Horvath - amdata.var.age
sns.jointplot(data=amdata.var, x='age', y='acc', hue='status')
sns.scatterplot(data=amdata.var, x='Horvath_acc', y='acc')
sns.scatterplot(data=amdata.var, x='age', y='acc', hue='status')
sns.histplot(data=amdata.var, x='status', hue='sex')
amdata.var.status.value_counts()

amdata.var[amdata.var.acc>0.8]
meta.loc[['GSM1272134', 'GSM1272153']]['characteristics_ch1.2']
sns.scatterplot(data=amdata.var, x='acc', y='bias', hue='status')
sns.jointplot(data=amdata.var, x='age', y='bias', hue='status')