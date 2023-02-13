#%% imports
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
import arviz as az

from sklearn.linear_model import LinearRegression

amdata = ad.read_h5ad('../exports/wave3_meta.h5ad', backed='r')
amdata_bio = ad.read_h5ad('../exports/wave3_acc.h5ad')

horvath_coef = pd.read_csv('../resources/Horvath_coef.csv', index_col=0)
intersection = amdata.obs.index.intersection(horvath_coef.index)
amdata_horvath = amdata[intersection].to_memory()

#%% compute accelerations
global_offsets = np.array([-0.1, -0.05, 0, 0.05, 0.1])

mod=LinearRegression().fit(X=amdata_horvath.X.T, y=amdata_horvath.var.age)
amdata_horvath.var['new_Horvath']=mod.predict(amdata_horvath.X.T)-amdata.var.age
sns.scatterplot(data=amdata_horvath.var, x='Horvath', y='new_Horvath')

accs =[]
horvaths = []

for offset in tqdm(global_offsets):
    amdata_bio_offset = amdata_bio.copy()
    amdata_horvath_offset = amdata_horvath.copy()
    amdata_bio_offset.X = amdata_bio.X+offset
    amdata_horvath_offset.X = amdata_horvath.X+offset
    maps = modelling_bio.person_model(amdata_bio_offset, return_trace=False, return_MAP=True, show_progress=False, cores=4)['map']
    accs.append(maps['acc'])
    horvaths.append((mod.predict(amdata_horvath_offset.X.T)-amdata_horvath.var.age).values)

#%% transform output
accs_df=pd.DataFrame(accs, index=np.round(global_offsets,2)).reset_index().melt(id_vars=['index'], var_name='offset').assign(type='acc')
horvaths_df=pd.DataFrame(horvaths, index=np.round(global_offsets,2)).reset_index().melt(id_vars=['index'], var_name='offset').assign(type='horvath')

accs_df['std'] = accs_df.groupby('index')['value'].transform('std')
accs_df['normalised_value'] = accs_df.value/accs_df['std']
horvaths_df['std'] = horvaths_df.groupby('index')['value'].transform('std')
horvaths_df['normalised_value'] = horvaths_df.value/horvaths_df['std']

df = pd.concat((accs_df, horvaths_df), axis=0)
df = df.drop(['offset'], axis=1).rename(columns={'index': 'offset'})

#%% plot
ax=sns.boxplot(data=df, x='offset', 
                y='normalised_value', hue='type', showfliers=False)

#%% save
plot.save(ax, 'A_Horvath_bias_problem', format='svg')
plot.save(ax, 'A_Horvath_bias_problem', format='png')