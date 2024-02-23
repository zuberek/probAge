#%% imports
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio_beta as modelling_bio
import arviz as az

import pickle

amdata = ad.read_h5ad('../exports/wave3_person_fitted.h5ad')

participants = amdata.var
participants = pd.read_csv('../exports/wave3_participants.csv', index_col=0)
# participants = participants.reset_index().set_index('name')

# import plotly.express as px
# px.scatter(data_frame=participants, x='acc', y='bias', hover_name=participants.index)

person_indexes = [
    '202915590012_R02C01',  # top negative bias
    '203041550151_R08C01',  # top positive bias
    '203141320012_R06C01',  # top negative acc
    '202939320015_R05C01',  # top positive acc
]

person_indexes = ['202939390027_R01C01', '202915420140_R04C01',
       '202939390103_R08C01', '202944920054_R03C01',
       '203068760066_R08C01', '203041550013_R04C01']

person_names = [
    7217,
    113186,
    6992,
    62845,
    9177,
]

person_names = [
    18750,
    137602,
    165342,
    39615,
    128692,
    24829,
]

#%% compoute traces
traces = []
for person_index in person_names:
    traces.append(modelling_bio.person_model(amdata[:,participants.loc[person_index]['index']], 
                                             method='nuts', progressbar=True))

with open('../exports/6_person_traces.idata', 'wb') as file:
    pickle.dump(traces, file)

#%% extract traces
with open('../exports/6_person_traces.idata', 'rb') as file:
    traces = pickle.load(file)
extracted_df = pd.DataFrame()
for trace in traces:
    tempdf = pd.DataFrame()
    tempdf['acc']= az.extract(trace.posterior).acc.values[0]
    tempdf['bias']= az.extract(trace.posterior).bias.values[0]
    tempdf['person'] = trace.posterior.part.values[0]
    extracted_df =pd.concat([extracted_df, tempdf], axis=0)

# %%
    
maps = modelling_bio.person_model(amdata[:, person_indexes], progressbar=True)
participants.loc[person_indexes, 'acc'] = maps['acc']
participants.loc[person_indexes, 'bias'] = maps['bias']

#%% plot
# colour plot
# ax = plot.row('Example participants acceleration and bias posteriors')
# sns.scatterplot(data=extracted_df, y='bias', x='acc', ax=ax,
#                 marker='.', alpha=0.3, hue='person', legend=False)
# sns.scatterplot(data=participants.loc[person_names].reset_index(), x='acc', y='bias', 
#                 hue='index', legend=False, ax=ax)                
# sns.kdeplot(data=extracted_df, y='bias', x='acc', hue='person', ax=ax,
#                 legend=False, fill=False)


maps= participants.loc[person_indexes][['acc', 'bias']]
plot.fonts(8)
g = sns.JointGrid(marginal_ticks=True,)
g.figure.set_size_inches((plot.cm2inch(9.5),plot.cm2inch(6)))

# sns.scatterplot(data=extracted_df, y='bias', x='acc', ax=g.ax_joint,
#                 marker='.', alpha=0.3, hue='person', legend=False)
sns.scatterplot(data=participants.loc[person_indexes].reset_index(), x='acc', y='bias', 
                hue='index', legend=False, ax=g.ax_joint, size=15)                
sns.kdeplot(data=extracted_df, levels=5, y='bias', x='acc', hue='person', 
                ax=g.ax_joint, thresh=0.25, legend=False, fill=False, linewidths=1)
sns.kdeplot(data=extracted_df, levels=5,  x='acc', common_norm=True, legend=False, 
            hue='person',ax=g.ax_marg_x, linewidth=1)
sns.kdeplot(data= extracted_df, levels=5, y='bias', common_norm=True, legend=False, 
            hue='person',ax=g.ax_marg_y, linewidth=1)
g.ax_marg_x.vlines(maps.values[:,0], ymin=0, ymax=2, colors=colors)
g.ax_marg_y.hlines(maps.values[:,1], xmin=0, xmax=40, colors=colors)
g.ax_joint.set_ylabel(r'Bias ($\beta$-value)')
g.ax_joint.set_xlabel('Acceleration \n' + r'($\beta$-value/year)')

# g.figure.tight_layout()

#%% plot
g.savefig('../figures/fig3/3B_kde_multi_person_posterior.svg')
g.savefig('../figures/fig3/3B_kde_multi_person_posterior.png')


