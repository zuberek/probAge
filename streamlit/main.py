import numpy as np
import pandas as pd
import anndata as ad
import streamlit as st

from src import modelling_bio
import arviz as az
import plotly.express as px
from scipy.stats import f_oneway
from streamlit_plotly_events import plotly_events

import plotly.io as pio
pio.templates.default = "plotly"


'# ProbAge'
amdata = None
person_index = None
selected_points = []
use_default = st.checkbox('Use the default downsyndrome dataset')

if use_default:
    amdata_path = '../resources/downsyndrome.h5ad'

    @st.cache_data
    def load_amdata():
        amdata = ad.read_h5ad(amdata_path)
        amdata.var.status= amdata.var.status.astype('str')
        amdata.var.loc[amdata.var.status=='Down syndrome', 'status'] = 'disease'
        amdata.var[['acc','bias']] = amdata.var[['acc','bias']].astype('float').round(3)
        return amdata

    amdata = load_amdata()

    # amdata

tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Compute", "Analyse dset", 'Analyse person'])

with tab1:

    site_info_path = '../resources/wave3_acc_sites.csv' 

    @st.cache_data
    def load_site_info():
        site_info = pd.read_csv(site_info_path, index_col=0)
        params = modelling_bio.get_site_params()
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

        st.plotly_chart(fig,theme=None)

        # selected_points = plotly_events(fig)
        
        # selected_points
        # selected_points[0]['x']
        if len(selected_points)>0:
            mask = amdata.var['acc']==selected_points[0]['x']
            person_index = amdata.var.iloc[np.nonzero(mask.values)[0][0]].name
            df=amdata.var.loc[person_index][:5]
            # df[['acc','bias']] = df[['acc','bias']].astype('float').round(2)
            df
    else:
        'Upload a dataset or use the default downsyndrome dataset'

with tab4:

    if amdata is not None:

        if use_default: f'**Using the default downsyndrome dataset of {amdata.shape[1]}**'
        'Person index: ', person_index


        if st.button('Reset selected person'):
            person_index = None

        '---'

        selection = st.selectbox(label='Select participant', options=amdata.var.index)
        person_index = selection

        if person_index is not None:
            f'Analysing the person **{person_index}**'
            amdata.var.loc[person_index]

            if st.button('Compute the posterior distributtion'):
                
                @st.cache_data
                def compute_trace(person_index):
                    return modelling_bio.person_model(amdata=amdata[:, person_index],
                        return_trace=True, return_MAP=False, show_progress=True)['trace']
                trace=compute_trace(person_index)

                trace.posterior.part.values

                st.pyplot(az.plot_pair(trace,kind='kde').get_figure())
            

    else:
        'Upload a dataset or use the default downsyndrome dataset'

# person_index = 'GSM1272194'
# amdata.var.loc[person_index]
# trace = modelling_bio.person_model(amdata=amdata[:, person_index],
#                         return_trace=True, return_MAP=False, show_progress=True)['trace']
