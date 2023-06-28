# %% 
# IMPORTS
import numpy as np
import pandas as pd
import anndata as ad
import pymc as pm

import streamlit as st
import arviz as az
import seaborn as sns
import plotly.express as px
from scipy.stats import f_oneway
from streamlit_plotly_events import plotly_events
import plotly.io as pio
pio.templates.default = "plotly"
from PIL import Image
import contextlib 
from sklearn.feature_selection import r_regression

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import src.modelling_bio  as modelling_bio
import src.preprocess_func  as preprocess_func

from modules.loading import test_limits, upload_data
from modules import loading, inferring
import modules

im = Image.open('streamlit/favicon.ico')
site_info_path = 'resources/wave3_acc_sites.csv' 
st.set_page_config(page_title='ProbBioAge', page_icon=im)
# st.set_page_config(page_title='ProbBioAge',layout='wide')


# %% 
# LOAD DATA
amdata = None
person_index = None
data = None
meta = None
site_meta = None
selected_points = []
trace = None

@st.cache_data
def load_site_info():
    site_info = pd.read_csv(site_info_path, index_col=0)
    params = modelling_bio.get_site_params()
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

test_limits()


st.warning('This is a protype version of the app. There very might be bugs which we are really sorry about! If you find any, please send an email to j.k.dabrowski@ed.ac.uk')



if st.checkbox('Analyse the default downsyndrome dataset'):
    st.session_state.DEFAULTED = True
    st.session_state.INFERRED = False
    st.session_state.UPLOADED = False
    st.session_state.TRAINED = False
    st.session_state.SELECTED = False
    st.session_state.TRACED = False
else:
    st.session_state.DEFAULTED = False

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
        amdata.var[['acc','bias']] = amdata.var[['acc','bias']].astype('float').round(3)
        return amdata

    amdata = load_def_amdata()



if st.session_state.DEFAULTED:
    tab_default = st.tabs(['**Analyse**'])
    tab1, tab2, tab3, tab4 = contextlib.nullcontext(), contextlib.nullcontext(), contextlib.nullcontext(), contextlib.nullcontext()
if not st.session_state.DEFAULTED:
    # tab_default = contextlib.nullcontext()
    tab1, tab2, tab3, tab4 = st.tabs(['**Upload**', '**Train**', '**Infer**', '**Analyse**'])



if not st.session_state.DEFAULTED:
    with tab1:

        tab1_status = st.container()

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



        # @st.cache_data
        # def load_uploaded(uploaded_file):
        #     meta= pd.read_csv(uploaded_file, index_col=0)
        #     if 'status' in meta.columns:
        #         meta.status = meta.status.astype('str')
        #     return meta
        

        # '### **Upload your methylation data**'
        # with st.expander('Expand for more information on the correct format'):
        #     st.markdown("""
        #         - All methylation values should be converted to beta values
        #         - Each row should correspond to a CpG site, each columns to a participant
        #         - The first column should be an index column with CpG site names
        #         - The first row should be an index row with participant indexes
                
        #         <br>
        #         Correctly formated dataset should look like this after loading:
        #     """, unsafe_allow_html=True)
        #     st.image("streamlit/correct_data_format.png")

        # uploaded_file1 = st.file_uploader(label='Upload your dataset',label_visibility='collapsed', type=['csv'])
        # if uploaded_file1 is not None:
        #     data = load_uploaded(uploaded_file1)
        #     'File dimensions: ', data.shape
        #     # st.dataframe(data.iloc[:5,:5])

        ''
        ''

        # '### **Upload your participant metadata**'
        # with st.expander('Expand for more information on the correct format'):
        #     st.markdown("""
        #         - Each row should correspond to a participant, each column to distinct information
        #         - The first column should be an index column with participant indexes
        #         - The file should at least contain columns named exactly 'age' and 'status'
        #             - Other columns are allowed and can be used in the downstream analysis
        #         - The 'age' column shoud contain participant age in years (can be a float)
        #         - The 'status' should have a string value of either 'control' standing for healthy individual or 'test' for an individual with a disease
                
        #         <br>
        #         Correctly formated dataset should look something like this after loading:
        #     """, unsafe_allow_html=True)
        #     st.image("streamlit/correct_metadata_format.png")
        # uploaded_file2 = st.file_uploader(label='Upload your metadata',label_visibility='collapsed', type=['csv'])
        # if uploaded_file2 is not None:
        #     meta = load_uploaded(uploaded_file2)
        #     correct = True

        #     if 'age' not in meta.columns:
        #         st.error("The metadata does not have 'age' column!")
        #         correct = False

        #     if 'status' not in meta.columns:
        #         st.error("The metadata does not have 'status' column!")
        #         correct = False

        #     if (~meta.status.isin(['control', 'test'])).any():
        #         st.error("The metadata 'status' column has values different than 'control' or 'test'!")
        #         correct = False

        #     if correct:
        #         'File dimensions: ', meta.shape
        #         # st.dataframe(meta.sample(5).iloc[:,:3])
        #     else:
        #         meta = None

        # ''
        # ''

        # if data is not None and meta is not None:

        #     @st.cache_data
        #     def load_amdata(data, meta):
        #         t = st.empty()
        #         t.markdown('Reading the data and metadata... ')
        #         try:
        #             amdata = ad.AnnData(X= data.values,
        #                     dtype=np.float32,
        #                     obs= pd.DataFrame(index=data.index),
        #                     var= meta)
                    
        #         except Exception as e:
        #             st.error("Error loading the data. Make sure it's in the correct format")
        #             with st.expander('Expand error log'):
        #                 e
        #         t.markdown('Reading the data and metadata ✅')

        #         t = st.empty()
        #         t.markdown('Preprocessing... ')
        #         amdata = preprocess_func.drop_nans(amdata)
        #         amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
        #         amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)
        #         t.markdown('Preprocessing ✅')
        #         return amdata
            
        #     amdata=load_amdata(data, meta)
        #     st.session_state.UPLOADED = True

        #     with tab1_status:
        #         st.success('Data is succesfully uploaded. You can move to the next tab', icon="✅")


        # ''
        # ''




    # with tab2:

    #     tab2_status = st.container()

    #     if not st.session_state.UPLOADED:
    #         st.warning('Upload the data and the metadata to run the training.')
    #     else:
    #         st.warning("""
    #         **You do not need to run the training.** Our approach includes a robust batch correction
    #         that utilizes the information learnt from the Generation Scotland dataset. 
    #         You can go to the next tab to infer the participant data.
    #         """)

    #         st.warning("""
    #         You can also upload site parameters if you have already trained the model.
    #         """)

    #         # uploaded_file3 = st.file_uploader(label='Upload your site parameters',label_visibility='collapsed', type=['csv'])
    #         # if uploaded_file3 is not None:
    #         #     site_meta = load_uploaded(uploaded_file3)
    #         #     'File dimensions: ', site_meta.shape
    #         #     amdata = amdata[site_meta.index]
    #         #     amdata.obs = site_meta
    #         #     st.session_state.TRAINED = True
    #         #     with tab2_status:
    #         #         st.success('Training finished succesfully. You can move to the next tab', icon="✅")
            
    #         if not st.session_state.TRAINED:

    #             '---'

    #             '### Site model training'

    #             from sklearn.feature_selection import r_regression

    #             @st.cache_data
    #             def training(_amdata, r2_threshold):

    #                 t = st.empty()
    #                 t.markdown('Computing R2 for every site... ')
    #                 my_bar = st.progress(0)
    #                 r2_array = []
    #                 for i, site_index in enumerate(_amdata.obs.index):
    #                     my_bar.progress(i/_amdata.n_obs)
    #                     r2_array.append(r_regression(_amdata[site_index].X.T, _amdata.var.age)[0]**2)
    #                 _amdata.obs['r2'] = r2_array
    #                 my_bar.empty()
    #                 t.markdown('Computing R2 for every site ✅')

    #                 selected, dropped = (_amdata.obs.r2>r2_threshold).value_counts().values
    #                 _amdata = _amdata[_amdata.obs.r2>r2_threshold]
    #                 'Selected ', selected, ' sites based on the r2 threshold (dropped ', dropped, ' sites)'

    #                 t = st.empty()
    #                 t.markdown('Computing MAPs for site parameters... ')
    #                 site_maps = modelling_bio.bio_sites(_amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
    #                 for param in modelling_bio.get_site_params():
    #                     _amdata.obs[param] = site_maps[param]
    #                 t.markdown('Computing MAPs for site parameters ✅')


    #                 _amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(_amdata, t=90)
    #                 _amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(_amdata)
    #                 _amdata.obs['saturating_der'] = _amdata.obs.abs_der<0.001

    #                 _amdata.obs['saturating'] = _amdata.obs.saturating_std | _amdata.obs.saturating_der

    #                 selected, dropped = _amdata.obs.saturating.value_counts().values
    #                 'Selected ', selected, ' sites based on the saturation threshold (dropped ', dropped, ' sites)'
    #                 # amdata = amdata[~amdata.obs.saturating]
    #                 # 'Selected ', selected, ' sites based on the r2 threshold (dropped ', dropped, ' sites)'

    #                 return _amdata

    #             r2_threshold = st.slider('R2 threshold for sites that are selected for training', 
    #                                     min_value=0.0, max_value=1.0, value=0.2)
    #             if st.button('Run the training'):
    #                 amdata=training(amdata, r2_threshold)

    #                 amdata

    #                 f'Download the learnt site parameters for {amdata.shape[0]} sites'
    #                 st.download_button(
    #                     label="⇩ Download CSV",
    #                     data=amdata.obs.to_csv().encode('utf-8'),
    #                     file_name='ProbBioAge_sites.csv',
    #                     mime='text/csv',
    #                 )

    #                 st.session_state.TRAINED = True

    #                 with tab2_status:
    #                     st.success('Training finished succesfully. You can move to the next tab', icon="✅")



    with tab3:

        tab3_status = st.container()

        if not st.session_state.UPLOADED:
            st.warning('Upload the data and the metadata to run the batch correction.')
        else:

            def revert_inference():
                st.session_state.INFERRED=False

            st.warning("""Select whether you want to infer participant data by retraining the model on your dataset, 
                        or the batch corrected Generation Scotland model.""")
            status = st.radio(label='Inference type', on_change=revert_inference,
                                options=['Infer using the trained model','Infer using the corrected GS model'],
                                label_visibility='collapsed')

            # @st.cache_data
            # def batch_correction(_amdata, status=st.session_state.INFERRED):

            #     t = st.empty()
            #     t.markdown('Preprocessing... ')
            #     intersection = site_info.index.intersection(_amdata.obs.index)
            #     _amdata = _amdata[intersection]
            #     _amdata.obs[params + ['r2']] = site_info[params + ['r2']]
            #     t.markdown('Preprocessing ✅')

            #     t = st.empty()
            #     t.markdown('Inferring site offsets... ')
            #     maps = modelling_bio.site_offsets(_amdata[:,_amdata.var.status=='control'], return_MAP=True, return_trace=False, show_progress=True)['map']
            #     _amdata.obs['offset'] = maps['offset']
            #     _amdata = _amdata[_amdata.obs.sort_values('offset').index]
            #     t.markdown('Inferring site offsets ✅')

            #     t = st.empty()
            #     t.markdown('Inferring participant accelerations and biases... ')
            #     _amdata.obs.eta_0 = _amdata.obs.eta_0 + _amdata.obs.offset
            #     _amdata.obs.meth_init  = _amdata.obs.meth_init + _amdata.obs.offset
            #     ab_maps = modelling_bio.person_model(_amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
            #     _amdata.var['acc'] = ab_maps['acc']
            #     _amdata.var['bias'] = ab_maps['bias']
            #     t.markdown('Inferring participant accelerations and biases ✅')
            #     return _amdata
            
            # if status=='Infer using the trained model':
            #     '---'

            #     '### Model training'

            #     st.warning("""
            #     You can also upload site parameters if you have already trained the model.
            #     """)

            #     uploaded_file3 = st.file_uploader(label='Upload your site parameters',label_visibility='collapsed', type=['csv'])
            #     if uploaded_file3 is not None:
            #         site_meta = load_uploaded(uploaded_file3)
            #         'File dimensions: ', site_meta.shape
            #         amdata = amdata[site_meta.index]
            #         amdata.obs = site_meta
            #         st.session_state.TRAINED = True

            #     r2_threshold = st.slider('R2 threshold for sites that are selected for training', 
            #             min_value=0.0, max_value=1.0, value=0.2)
            
            # if st.button('Run the inference'):
            #     inference(status, amdata)
                
            # @st.cache_data
            # def inference(status, _amdata):

            #     if status=='Infer using the corrected GS model':
            #         if st.button('Run the inference'):
            #             with col1:
            #                 amdata = batch_correction(amdata)
            #                 'Dataset: ', amdata
            #             with col2:
            #                 st.pyplot(sns.histplot(amdata.obs.offset).get_figure())
            #             st.session_state.INFERRED = True

            #     if status=='Infer using the trained model':
            #         @st.cache_data
            #         def computing_r2(_amdata):

            #             t = st.empty()
            #             t.markdown('Computing R2 for every site... ')
            #             my_bar = st.progress(0)
            #             r2_array = []
            #             for i, site_index in enumerate(_amdata.obs.index):
            #                 my_bar.progress(i/_amdata.n_obs)
            #                 r2_array.append(r_regression(_amdata[site_index].X.T, _amdata.var.age)[0]**2)
            #             _amdata.obs['r2'] = r2_array
            #             my_bar.empty()
            #             t.markdown('Computing R2 for every site ✅')
            #             return _amdata

            #         @st.cache_data
            #         def retraining(_amdata):
            #             _amdata.var

            #             selected, dropped = (_amdata.obs.r2>r2_threshold).value_counts().values
            #             _amdata = _amdata[_amdata.obs.r2>r2_threshold]
            #             'Selected ', selected, ' sites based on the r2 threshold (dropped ', dropped, ' sites)'

            #             t = st.empty()
            #             t.markdown('Computing MAPs for site parameters... ')
            #             site_maps = modelling_bio.bio_sites(_amdata[:, _amdata.var.status=='control'], return_MAP=True, return_trace=False, show_progress=True)['map']
            #             for param in modelling_bio.get_site_params():
            #                 _amdata.obs[param] = site_maps[param]
            #             t.markdown('Computing MAPs for site parameters ✅')

            #             _amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(_amdata, t=90)
            #             _amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(_amdata)
            #             _amdata.obs['saturating_der'] = _amdata.obs.abs_der<0.001

            #             _amdata.obs['saturating'] = _amdata.obs.saturating_std | _amdata.obs.saturating_der
            #             return _amdata

            #         @st.cache_data
            #         def inferring(_amdata):

            #             selected, dropped = _amdata.obs.saturating.value_counts().values
            #             'There are ', selected, ' saturating sites sites (', dropped, ' not saturating sites)'
            #             _amdata = _amdata[~_amdata.obs.saturating]

            #             if _amdata.n_obs>250:
            #                 'Selecting best 250 r2 CpG sites'
            #                 _amdata=_amdata[_amdata.obs.sort_values('r2').tail(250).index]

            #             t = st.empty()
            #             t.markdown('Inferring participant accelerations and biases... ')
            #             ab_maps = modelling_bio.person_model(_amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
            #             _amdata.var['acc'] = ab_maps['acc']
            #             _amdata.var['bias'] = ab_maps['bias']
            #             t.markdown('Inferring participant accelerations and biases ✅')
            #             return _amdata

            #         col1, col2 = st.columns(2)

            #         with col1:
            #             amdata = computing_r2(amdata)
            #         with col2:
            #             ax=sns.histplot(amdata.obs.r2)
            #             ax.axvline(x=r2_threshold, ymin=0, ymax=1)
            #             st.pyplot(ax.get_figure())

            #         with col1:
            #             amdata = retraining(amdata)
            #             site_params = amdata.obs
            #             amdata = inferring(amdata)                  
            #             'Dataset: ', amdata

            #         f'Download the learnt site parameters for {site_params.shape[0]} sites'
            #         st.download_button(
            #             label="⇩ Download CSV",
            #             data=site_params.to_csv().encode('utf-8'),
            #             file_name='ProbBAge_sites.csv',
            #             mime='text/csv',
            #         )



            #         st.session_state.INFERRED = True
            

            # if status=='Infer using the trained model':

            #     '---'

            #     '### Model training'

            #     st.warning("""
            #     You can also upload site parameters if you have already trained the model.
            #     """)

            #     uploaded_file3 = st.file_uploader(label='Upload your site parameters',label_visibility='collapsed', type=['csv'])
            #     if uploaded_file3 is not None:
            #         site_meta = load_uploaded(uploaded_file3)
            #         'File dimensions: ', site_meta.shape
            #         amdata = amdata[site_meta.index]
            #         amdata.obs = site_meta
            #         st.session_state.TRAINED = True

            #     r2_threshold = st.slider('R2 threshold for sites that are selected for training', 
            #             min_value=0.0, max_value=1.0, value=0.2)

            #     if st.button('Run the inference'):

            #         @st.cache_data
            #         def computing_r2(_amdata):

            #             t = st.empty()
            #             t.markdown('Computing R2 for every site... ')
            #             my_bar = st.progress(0)
            #             r2_array = []
            #             for i, site_index in enumerate(_amdata.obs.index):
            #                 my_bar.progress(i/_amdata.n_obs)
            #                 r2_array.append(r_regression(_amdata[site_index].X.T, _amdata.var.age)[0]**2)
            #             _amdata.obs['r2'] = r2_array
            #             my_bar.empty()
            #             t.markdown('Computing R2 for every site ✅')
            #             return _amdata

            #         @st.cache_data
            #         def retraining(_amdata):
            #             selected, dropped = (_amdata.obs.r2>r2_threshold).value_counts().values
            #             _amdata = _amdata[_amdata.obs.r2>r2_threshold]
            #             'Selected ', selected, ' sites based on the r2 threshold (dropped ', dropped, ' sites)'

            #             t = st.empty()
            #             t.markdown('Computing MAPs for site parameters... ')
            #             site_maps = modelling_bio.bio_sites(_amdata[:, _amdata.var.status=='control'], return_MAP=True, return_trace=False, show_progress=True)['map']
            #             for param in modelling_bio.get_site_params():
            #                 _amdata.obs[param] = site_maps[param]
            #             t.markdown('Computing MAPs for site parameters ✅')

            #             _amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(_amdata, t=90)
            #             _amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(_amdata)
            #             _amdata.obs['saturating_der'] = _amdata.obs.abs_der<0.001

            #             _amdata.obs['saturating'] = _amdata.obs.saturating_std | _amdata.obs.saturating_der
            #             return _amdata

            #         @st.cache_data
            #         def inferring(_amdata):

            #             selected, dropped = _amdata.obs.saturating.value_counts().values
            #             'There are ', selected, ' saturating sites sites (', dropped, ' not saturating sites)'
            #             _amdata = _amdata[~_amdata.obs.saturating]

            #             if _amdata.n_obs>250:
            #                 'Selecting best 250 r2 CpG sites'
            #                 _amdata=_amdata[_amdata.obs.sort_values('r2').tail(250).index]

            #             t = st.empty()
            #             t.markdown('Inferring participant accelerations and biases... ')
            #             ab_maps = modelling_bio.person_model(_amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
            #             _amdata.var['acc'] = ab_maps['acc']
            #             _amdata.var['bias'] = ab_maps['bias']
            #             t.markdown('Inferring participant accelerations and biases ✅')
            #             return _amdata

            #         col1, col2 = st.columns(2)

            #         with col1:
            #             amdata = computing_r2(amdata)
            #         with col2:
            #             ax=sns.histplot(amdata.obs.r2)
            #             ax.axvline(x=r2_threshold, ymin=0, ymax=1)
            #             st.pyplot(ax.get_figure())

            #         with col1:
            #             amdata = retraining(amdata)
            #             site_params = amdata.obs
            #             amdata = inferring(amdata)                  
            #             'Dataset: ', amdata

            #         f'Download the learnt site parameters for {site_params.shape[0]} sites'
            #         st.download_button(
            #             label="⇩ Download CSV",
            #             data=site_params.to_csv().encode('utf-8'),
            #             file_name='ProbBAge_sites.csv',
            #             mime='text/csv',
            #         )



            #         st.session_state.INFERRED = True


            # if status=='Infer using the corrected GS model':
            #     if st.button('Run the inference'):
            #         col1, col2 = st.columns(2)
            #         with col1:
            #             # amdata = batch_correction(amdata)
            #             amdata = modules.inferring.batch_correction(amdata)
            #             st.write('Dataset: ', amdata)
            #         with col2:
            #             st.pyplot(sns.histplot(amdata.obs.offset).get_figure())
            #         st.session_state.INFERRED = True

            #             

            if status=='Infer using the corrected GS model':
                if st.button('Run the inference'):
                    amdata = modules.inferring.batch_correction(amdata)
            
            if status=='Infer using the trained model':
                amdata = modules.inferring.model_retraining(amdata)


            '---'

            if st.session_state.INFERRED:

                # f'Download the acceleration and bias for {amdata.var.shape[0]} participants'
                # st.download_button(
                #     label="⇩ Download CSV",
                #     data=amdata.var[['acc','bias']].to_csv().encode('utf-8'),
                #     file_name='ProbBioAge_results.csv',
                #     mime='text/csv',
                # )

                

                with tab3_status:
                    st.success('Inference finished succesfully. You can move to the next tab', icon="✅")





with tab4:
    if st.session_state.INFERRED or st.session_state.DEFAULTED:
    # if amdata is not None:
        # 'Dataset: ', amdata

        # print_state()

        # def compute_anova():
        #     acc_control = amdata[:, amdata.var.status == 'control'].var['acc'].values
        #     acc_down = amdata[:, amdata.var.status == 'test'].var['acc'].values
        #     return f_oneway(acc_control, acc_down)
        # anova = compute_anova()
        # st.write(f'The ANOVA statistic is {round(anova[0],2)} and pvalue is {round(anova[1],2)}')
        

        col1, col2, col3 = st.columns(3)

        columns = amdata.var.columns
        X_axis = col1.selectbox("Choose X axis:", amdata.var.columns, index=amdata.var.columns.to_list().index('acc'))
        Y_axis = col2.selectbox("Choose Y axis:", amdata.var.columns,index=amdata.var.columns.to_list().index('bias'))
        color = col3.selectbox("Choose color column:", amdata.var.columns,index=amdata.var.columns.to_list().index('status'))
        

        plot_spot = st.empty()

        fig = px.scatter(data_frame=amdata.var, x=X_axis, y=Y_axis, color=color, 
                        marginal_x='box', marginal_y='box',hover_name=amdata.var.index)
        
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ))

        with plot_spot:
            selected_points = plotly_events(fig)
            # st.plotly_chart(fig,theme=None)
        # # st.plotly_chart(fig,theme=None)

        
        # selected_points
        # selected_points[0]['x']

            
        col1, col2 = st.columns(2)

        if len(selected_points)>0:
            mask = amdata.var['acc']==selected_points[0]['x']
            st.session_state.SELECTED =  amdata.var.iloc[np.nonzero(mask.values)[0][0]].name
            st.session_state.TRACED = False
        else:
            st.success('Click a point on the scatterplot to investigate a participant')


        if st.session_state.SELECTED is not False:
            person_index = st.session_state.SELECTED
            with col1:
                df=amdata.var.loc[person_index]
                df

            with col2:

                if st.button('Compute the posterior distributtion'):
                    st.session_state.TRACED = True

                if st.session_state.TRACED:
                    @st.cache_data
                    def compute_trace(person_index):
                        trace = modelling_bio.person_model(amdata=amdata[:, person_index],
                            return_trace=True, return_MAP=False, show_progress=True)['trace']
                        trace.posterior['acc'] = np.log2(trace.posterior.acc)
                        return trace
                    trace=compute_trace(person_index)

                    # trace.posterior.part.values

                    st.pyplot(az.plot_pair(trace,kind='kde').get_figure())
            
    else:
        st.warning('Either use the default downsyndrome dataset, or upload the data and the metadata and run the batch correction.')


if st.session_state.DEBUGGING:
    with st.sidebar:
        '**Cached data**'
        'data: ', data.shape if data is not None else data
        'meta: ', meta.shape if meta is not None else meta
        'amdata: ', amdata.shape if amdata is not None else amdata
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



# with tab4:

#     if amdata is not None:

#         if st.session_state.DEFAULTED: f'**Using the default downsyndrome dataset of {amdata.shape[1]}**'
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
