import numpy as np
import pandas as pd
import anndata as ad
import pymc as pm

import sys
sys.path.append("..")   # fix to import modules from root
import streamlit as st

import arviz as az
import plotly.express as px
from scipy.stats import f_oneway
from streamlit_plotly_events import plotly_events

import plotly.io as pio
pio.templates.default = "plotly"

# TODO Provide a list cpg sites to filter out if the uploaded dataset is too big

CHAINS = 4
CORES = 1
SITE_PARAMETERS = {
    'eta_0':    'eta_0',
    'omega':    'omega',
    'p':        'meth_init',
    'N':        'system_size',
    'var_init': 'var_init'
}

def get_site_params():
    return list(SITE_PARAMETERS.values())


def person_model(amdata,
                         return_trace=True,
                         return_MAP=False,
                         show_progress=False,
                         init_nuts='auto',
                         cores=CORES,
                         map_method='L-BFGS-B'):

    if show_progress: print(f'Modelling {amdata.shape[1]} participants')

    # The data has two dimensions: participant and CpG site
    coords = {"site": amdata.obs.index, "part": amdata.var.index}

    # # create a numpy array of the participants ages
    # # array of ages needs to be broadcasted into a matrix array for each CpG site
    omega = np.broadcast_to(amdata.obs.omega, shape=(amdata.shape[1], amdata.shape[0])).T
    eta_0 = np.broadcast_to(amdata.obs.eta_0, shape=(amdata.shape[1], amdata.shape[0])).T
    p = np.broadcast_to(amdata.obs.meth_init, shape=(amdata.shape[1], amdata.shape[0])).T
    var_init = np.broadcast_to(amdata.obs.var_init, shape=(amdata.shape[1], amdata.shape[0])).T
    N = np.broadcast_to(amdata.obs.system_size, shape=(amdata.shape[1], amdata.shape[0])).T

    age = amdata.var.age.values


    # Define Pymc model
    with pm.Model(coords=coords) as model:
        
        # Define model variables
        acc = pm.Uniform('acc', lower=0.33, upper=3, dims='part')
        bias = pm.Normal('bias', mu=0, sigma=0.1, dims='part')

        # Useful variables
        omega = acc*omega
        eta_1 = 1-eta_0
        
        # model mean and variance
        var_term_0 = eta_0*eta_1
        var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)


        # mean = eta_0 + np.exp(-omega*age)*((p-1)*eta_0 + p*eta_1)
        mean = eta_0 + np.exp(-omega*age)*((p-1)*eta_0 + p*eta_1) + bias

        variance = (var_term_0/N 
                + np.exp(-omega*age)*(var_term_1-var_term_0)/N 
                + np.exp(-2*omega*age)*(var_init/np.power(N,2) - var_term_1/N)
            )

        # Define likelihood
        obs = pm.Normal("obs",
                            mu=mean,
                            sigma = np.sqrt(variance), 
                            dims=("site", "part"), 
                            observed=amdata.X)

        res = {}
        if return_MAP:
            res['map'] = pm.find_MAP(progressbar=show_progress,
                                     method=map_method,
                                     maxeval=10_000)
            res['map']['acc'] = np.log2(res['map']['acc'])

        if return_trace:
            res['trace'] = pm.sample(1000, tune=1000, init=init_nuts,
                                    chains=CHAINS, cores=cores,
                                    progressbar=show_progress)

    return res    


'# ProbAge'
amdata = None
person_index = None
selected_points = []
use_default = st.checkbox('Use the default downsyndrome dataset')

if use_default:
    amdata_path = 'resources/downsyndrome.h5ad'

    @st.cache_data
    def load_amdata():
        amdata = ad.read_h5ad(amdata_path)
        amdata.var.status= amdata.var.status.astype('str')
        amdata.var.loc[amdata.var.status=='Down syndrome', 'status'] = 'disease'
        amdata.var[['acc','bias']] = amdata.var[['acc','bias']].astype('float').round(3)
        return amdata

    amdata = load_amdata()

    # amdata

tab1, tab2, tab3 = st.tabs(["Upload", "Compute", "Analyse"])

with tab1:

    site_info_path = 'resources/wave3_acc_sites.csv' 

    @st.cache_data
    def load_site_info():
        site_info = pd.read_csv(site_info_path, index_col=0)
        params = get_site_params()
        return site_info, params
    site_info, params = load_site_info()

    uploaded_file = st.file_uploader(label='Upload control dataset', type=['csv'])

    @st.cache_data
    def load_uploaded():
        return pd.read_csv(uploaded_file, index_col=0)
    if uploaded_file is not None:
        data = load_uploaded()

with tab3:

    'Amdata: ', amdata

    if amdata is not None:

        # if use_default: f'**Using the default downsyndrome dataset of {amdata.shape[1]}**'
        
        @st.cache_data
        def compute_anova():
            acc_control = amdata[:, amdata.var.status == 'healthy'].var['acc'].values
            acc_down = amdata[:, amdata.var.status == 'Down syndrome'].var['acc'].values
            return f_oneway(acc_control, acc_down)
        anova = compute_anova()
        st.write(f'The ANOVA statistic is {round(anova[0],2)} and pvalue is {round(anova[1],2)}')
        

        fig = px.scatter(data_frame=amdata.var, x='acc', y='bias', color='status', 
                        marginal_x='box', marginal_y='box',hover_name=amdata.var.index)

        # st.plotly_chart(fig,theme=None)

        selected_points = plotly_events(fig)
        
        # selected_points
        # selected_points[0]['x']

        col1, col2 = st.columns(2)

        with col1:
            if len(selected_points)>0:
                mask = amdata.var['acc']==selected_points[0]['x']
                person_index = amdata.var.iloc[np.nonzero(mask.values)[0][0]].name
                df=amdata.var.loc[person_index]
                # df[['acc','bias']] = df[['acc','bias']].astype('float').round(2)
                df

        with col2:
            if person_index is not None:
                f'Analysing the person **{person_index}**'
                # amdata.var.loc[person_index]

                if st.button('Compute the posterior distributtion'):
                    
                    @st.cache_data
                    def compute_trace(person_index):
                        return person_model(amdata=amdata[:, person_index],
                            return_trace=True, return_MAP=False, show_progress=True)['trace']
                    trace=compute_trace(person_index)

                    # trace.posterior.part.values

                    st.pyplot(az.plot_pair(trace,kind='kde').get_figure())
            
    else:
        'Upload a dataset or use the default downsyndrome dataset'

# with tab4:

#     if amdata is not None:

#         if use_default: f'**Using the default downsyndrome dataset of {amdata.shape[1]}**'
#         'Person index: ', person_index


#         # if st.button('Reset selected person'):
#         #     person_index = None

#         # '---'

#         selection = st.selectbox(label='Select participant', options=amdata.var.index)
#         person_index = selection

#         if person_index is not None:
#             f'Analysing the person **{person_index}**'
#             amdata.var.loc[person_index]

#             if st.button('Compute the posterior distributtion'):
                
#                 @st.cache_data
#                 def compute_trace(person_index):
#                     return person_model(amdata=amdata[:, person_index],
#                         return_trace=True, return_MAP=False, show_progress=True)['trace']
#                 trace=compute_trace(person_index)

#                 trace.posterior.part.values

#                 st.pyplot(az.plot_pair(trace,kind='kde').get_figure())
            

#     else:
#         'Upload a dataset or use the default downsyndrome dataset'

# person_index = 'GSM1272194'
# amdata.var.loc[person_index]
# trace = person_model(amdata=amdata[:, person_index],
#                         return_trace=True, return_MAP=False, show_progress=True)['trace']
