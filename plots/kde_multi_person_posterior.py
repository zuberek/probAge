%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling_bio
import arviz as az

amdata = ad.read_h5ad('../exports/wave3_acc.h5ad')

person_indexes = [
    '202915590012_R02C01',  # top negative bias
    '203041550151_R08C01',  # top positive bias
    '203141320012_R06C01',  # top negative acc
    '202939320015_R05C01',  # top positive acc
]

traces = []
for person_index in person_indexes:
    traces.append(modelling_bio.person_model(amdata[:,person_index], return_trace=True, cores=4,
                            return_MAP=False, show_progress=True, )['trace'])

extracted_df = pd.DataFrame()
for trace in traces:
    tempdf = pd.DataFrame()
    tempdf['acc']= az.extract(trace.posterior).acc.values[0]
    tempdf['bias']= az.extract(trace.posterior).bias.values[0]
    tempdf['person'] = trace.posterior.part.values[0]
    extracted_df =pd.concat([extracted_df, tempdf], axis=0)


ax = sns.kdeplot(data=extracted_df, x='bias', y='acc', hue='person')

plot.save(ax, 'kde_multi_person_posterior', format='svg')
plot.save(ax, 'kde_multi_person_posterior', format='png')
