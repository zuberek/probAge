import streamlit as st
import plotly.express as px
import seaborn as sns
import numpy as np
import arviz as az
import pandas as pd
from modules import loading
from src import modelling_bio_beta as modelling
import anndata as ad

from streamlit_plotly_events import plotly_events

if 'SELECTED' not in st.session_state:
    st.session_state.SELECTED = False
if 'TRACED' not in st.session_state:
    st.session_state.TRACED = False

site_info_path = 'resources/ewas_fitted_sites.csv' 

if 'SITE_INFO' not in st.session_state:
    st.session_state.SITE_INFO = pd.read_csv(site_info_path, index_col=0)
if 'PARAMS'  not in st.session_state:
    st.session_state.PARAMS = list(modelling.SITE_PARAMETERS.values())


if 'DATA' not in st.session_state:
    st.warning('Upload data and metadata to run model inference.')
    
    'You can also upload your resulting anndata inference file here to analyze it.'
    
    uploaded_h5ad = st.file_uploader(
        "Upload AnnData file",
        type="h5ad"
    )
    
    if uploaded_h5ad:
        amdata = ad.read_h5ad(uploaded_h5ad)
        required_cols = {"acc", "bias"}
        if not required_cols.issubset(amdata.var.columns):
            st.error("AnnData must contain 'acc' and 'bias' in .var")
            st.stop()

        st.session_state.DATA = amdata
        st.rerun()
else:
    amdata = st.session_state.DATA
    
    fig = px.scatter(data_frame=amdata.var, x='acc', y='bias', color='status', 
                marginal_x='box', marginal_y='box',hover_name=amdata.var.index)

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ))
    
    selected_points = st.plotly_chart(
        fig,
        width='stretch',
        on_select="rerun"
    )

    col1, col2 = st.columns(2)
    
    if selected_points and selected_points.get("selection"):
        points = selected_points["selection"]["points"]

        if len(points) > 0:
            idx = points[0]["hovertext"]
            st.session_state.SELECTED = idx

    else:
        st.success('Click a point on the scatterplot to investigate a participant (For now only works in the acc vs bias view)')


    if st.session_state.SELECTED is not False:
        person_index = st.session_state.SELECTED
        with col1:
            df=amdata.var.loc[person_index]
            # df[['acc','bias']] = df[['acc','bias']].astype('float').round(2)
            df

    with col2:
        if st.session_state.SELECTED is not False:
            person_index = st.session_state.SELECTED
            f'Analysing the person **{person_index}**'
            # amdata.var.loc[person_index]

            if st.button('Compute the posterior distributtion'):
                
                @st.cache_data
                def compute_trace(person_index):
                    return modelling.person_model(amdata=amdata[:, person_index],
                        method='nuts', progressbar=True)
                trace=compute_trace(person_index)

                # trace.posterior.part.values

                st.pyplot(az.plot_pair(trace,kind='kde').get_figure())

