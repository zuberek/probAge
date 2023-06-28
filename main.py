# %% 
# IMPORTS
import numpy as np
import pandas as pd
import anndata as ad
import pymc as pm

import sys
# sys.path.append("..")   # fix to import modules from root
import streamlit as st
import pathlib
# sys.path.append(str(pathlib.Path().absolute()).split("/streamlit")[0] + "/src")
import arviz as az
import plotly.express as px
from scipy.stats import f_oneway
from streamlit_plotly_events import plotly_events
sys.path.append('src')
# sys.path.append('src')
sys.path
import plotly.io as pio
from PIL import Image
pio.templates.default = "plotly"

import contextlib 

import src.modelling_bio  as modelling_bio
# from src.modelling_bio import person_model


# im = Image.open('streamlit/favicon.ico')
# site_info_path = 'resources/wave3_acc_sites.csv' 
site_info_path = 'resources/ewas_fitted_sites.csv' 
# st.set_page_config(page_title='ProbBioAge', page_icon=im)
# st.set_page_config(page_title='ProbBioAge',layout='wide')

from modules.loading import test_limits, upload_data
from modules import loading, inferring
import modules

# %% 
# LOAD DATA
amdata = None
# amdata2 = None
person_index = None
selected_points = []

@st.cache_data
def load_site_info():
    site_info = pd.read_csv(site_info_path, index_col=0)
    params = get_site_params()
    return site_info, params
site_info, params = load_site_info()

if 'SITE_INFO' not in st.session_state:
    st.session_state.SITE_INFO = pd.read_csv(site_info_path, index_col=0)
if 'PARAMS'  not in st.session_state:
    st.session_state.PARAMS = modelling_bio.get_site_params()

@st.cache_data
def convert_df(site_info):
    df = site_info.reset_index()['index'].to_frame()
    df.index.name = 'index'
    df = df.rename(columns={'index':'name'})
    return df.to_csv().encode('utf-8')

csv = convert_df(site_info)

# %% 
# STREAMLIT
'# ProbBioAge'


# states
if 'DEBUGGING' not in st.session_state:
    st.session_state.DEBUGGING = True
if 'TRAINED' not in st.session_state:
    st.session_state.TRAINED = False
if 'DEFAULTED' not in st.session_state:
    st.session_state.DEFAULTED = False
if 'UPLOADED' not in st.session_state:
    st.session_state.UPLOADED = False
if 'INFERRED' not in st.session_state:
    st.session_state.INFERRED = False
if 'SELECTED' not in st.session_state:
    st.session_state.SELECTED = False
if 'TRACED' not in st.session_state:
    st.session_state.TRACED = False

def print_state():
    'DEBUGGING: ', st.session_state.DEBUGGING
    'DEFAULTED: ', st.session_state.DEFAULTED
    'UPLOADED: ', st.session_state.UPLOADED
    'TRAINED: ', st.session_state.TRAINED
    'INFERRED: ', st.session_state.INFERRED
    'SELECTED: ', st.session_state.SELECTED
    'TRACED: ', st.session_state.TRACED


@st.cache_data
def switch_inferred():
    return True

'# ProBAge'

# test_limits()


st.warning('This is a protype version of the app. There very might be bugs which we are really sorry about! If you find any, please send an email to j.k.dabrowski@ed.ac.uk. If the app seems to have stopped working, try clearing the cache!')



if st.checkbox('Analyse the default downsyndrome dataset'):
    st.session_state.DEFAULTED = True
    st.session_state.INFERRED = False
    st.session_state.UPLOADED = False
    st.session_state.TRAINED = False
    st.session_state.SELECTED = False
    st.session_state.TRACED = False
else:
    st.session_state.DEFAULTED = False

if st.button('Clear cache (for rerunning)', help='Often solves problems when running the app'):
    st.cache_data.clear()

# amdata.var = amdata.var.rename(columns={'control':'status'})
# amdata.var.status = amdata.var.status.astype('O')
# amdata.var.loc[amdata.var.control=='healthy', 'control'] = 'True'
# # amdata.write_h5ad(amdata_path)
# amdata.var.loc[amdata.var.status=='False', 'status'] = 'test'
# amdata.var.loc[amdata.var.status=='True', 'status'] = 'control'
# amdata.write_h5ad(amdata_path)
if st.session_state.DEFAULTED:
    amdata_path = 'resources/downsyndrome.h5ad'

    @st.cache_data
    def load_def_amdata():
        amdata = ad.read_h5ad(amdata_path)
        amdata.var.status= amdata.var.status.astype('str')
        amdata.var.loc[amdata.var.status=='Down syndrome', 'status'] = 'disease'
        amdata.var[['acc','bias']] = amdata.var[['acc','bias']].astype('float').round(3)
        return amdata

    amdata = load_def_amdata()



if use_default:
    tab_default = st.tabs(['**Analyse**'])
    tab1, tab2, tab3 = contextlib.nullcontext(), contextlib.nullcontext(), contextlib.nullcontext()
if not st.session_state.DEFAULTED:
    # tab_default = contextlib.nullcontext()
    tab1, tab2, tab3 = st.tabs(['**Upload**', '**Infer**', '**Analyse**'])


###########################################
### TAB 1: UPLOADING
###########################################

if not use_default:
    with tab1:
        
        data = None
        meta = None

        data = modules.loading.upload_data()
        meta = modules.loading.upload_meta()
        amdata = modules.loading.create_anndata(data, meta)


        if st.session_state.UPLOADED:
            with tab1_status:
                st.success('Data is succesfully uploaded. You can move to the next tab', icon="✅")

        'Download a list of required CpG sites to filter your dataset (in case your dataset is too big)'
        st.download_button(
            label="⇩ Download CSV (3.6 kB)",
            data=csv,
            file_name='ProbBioAge_cpgs.csv',
            mime='text/csv',
        )

###########################################
### TAB 2: INFERRING
###########################################

    with tab2:

        tab2_status = st.container()

        if not st.session_state.UPLOADED:
            st.warning('Upload the data and the metadata to run the batch correction.')
        else:

            def revert_inference():
                st.session_state.INFERRED=False
                st.session_state.SELECTED=False

            st.warning("""Select whether you want to infer participant data by retraining the model on your dataset, 
                        or the batch corrected Generation Scotland model.""")
            status = st.radio(label='Inference type', on_change=revert_inference,
                                options=['Infer using our batch corrected pretrained model','Infer by retraining the model on your data'],
                                label_visibility='collapsed')


            if status=='Infer using our batch corrected pretrained model':
                if st.button('Run the batch correction'):
                    amdata = modules.inferring.batch_correction(status,amdata)
                    st.session_state.INFERRED = True



                
            if status=='Infer by retraining the model on your data':
                amdata = modules.inferring.model_retraining(status, amdata)



            '---'

            if st.session_state.INFERRED:

                @st.cache_data
                def cache_amdata(state=st.session_state.INFERRED):
                    amdata
                    return amdata
                

                # @st.cache_data
                # def cache_participants(state=st.session_state.INFERRED):
                #     return amdata.var
                # participants = cache_participants()





                with tab2_status:
                    st.success('Inference finished succesfully. You can move to the next tab', icon="✅")
                    amdata=cache_amdata()

                    f'Download the acceleration and bias for {amdata.var.shape[0]} participants'
                    st.download_button(
                        label="⇩ Download CSV",
                        data=amdata.var.to_csv().encode('utf-8'),
                        file_name='ProbBioAge_results.csv',
                        mime='text/csv',
                    )

                    '---'


###########################################
### TAB 2: ANALYSING
###########################################

 
with tab3:
    if st.session_state.INFERRED or st.session_state.DEFAULTED:

        @st.cache_data
        def cache_participants(_amdata, state=st.session_state.INFERRED):
            return _amdata.var
        participants = cache_participants(amdata)

        @st.cache_data
        def cache_amdata(state=st.session_state.INFERRED):
            return amdata
        amdata=cache_amdata()

        fig = px.scatter(data_frame=amdata.var, x='acc', y='bias', color='status', 
                        marginal_x='box', marginal_y='box',hover_name=amdata.var.index)

        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ))
        # st.plotly_chart(fig,theme=None)

        selected_points = plotly_events(fig)
        
        # selected_points
        # selected_points[0]['x']

        col1, col2 = st.columns(2)

        if len(selected_points)>0:
            mask = amdata.var['acc']==selected_points[0]['x']
            st.session_state.SELECTED =  amdata.var.iloc[np.nonzero(mask.values)[0][0]].name
            st.session_state.TRACED = False
        else:
            st.success('Click a point on the scatterplot to investigate a participant (For now only works in the acc vs bias view)')


        if st.session_state.SELECTED is not False:
            person_index = st.session_state.SELECTED
            with col1:
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
        st.warning('Either use the default downsyndrome dataset, or upload the data and the metadata and run the batch correction.')


###########################################
### DEBUGGING SIDEBAR
###########################################

if st.session_state.DEBUGGING:
    with st.sidebar:
        '**Cached data**'
        'data: ', data.shape if data is not None else data
        'meta: ', meta.shape if meta is not None else meta
        'amdata: ', amdata.shape if amdata is not None else amdata
        'amdata: ', amdata if amdata is not None else amdata
        # 'amdata2: ', amdata2 if amdata2 is not None else amdata2
        'person_index: ', person_index
        'trace: ', trace.posterior.dims if trace is not None else trace

        '---'

        '**State**'

        def test():
            st.session_state.UPLOADED= not uploaded
        uploaded = st.checkbox('UPLOADED', value=st.session_state.UPLOADED, on_change=test)
        print_state()

        '---'

        f'Download downsyndrome data (to debug the upload)'
        st.download_button(
            label="⇩ Methylation data (123kB)",
            data=open("streamlit/downsyndrome.csv", "r"),
            file_name='downsyndrome.csv',
            mime='text/csv',
        )

        st.download_button(
            label="⇩ Meta data (4kB)",
            data=open("streamlit/downsyndrome_meta.csv", "r"),
            file_name='downsyndrome_meta.csv',
            mime='text/csv',
        )


# Use <br> below to print the lines closer
''

st.markdown("""
<br><br><br>

---

For reference, check out our preprint: https://doi.org/10.1101/2023.03.01.530570 <br>
For code, check out our repository: https://github.com/zuberek/ProbBioAge
""", unsafe_allow_html=True
)
