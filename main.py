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

import src.modelling_bio  as modelling_bio
import src.preprocess_func  as preprocess_func

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
selected_points = []
trace = None

@st.cache_data
def load_site_info():
    site_info = pd.read_csv(site_info_path, index_col=0)
    params = modelling_bio.get_site_params()
    return site_info, params
site_info, params = load_site_info()

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

@st.cache_resource
def switch_inferred():
    return True

'# ProbBioAge'

if st.checkbox('Use the default downsyndrome dataset'):
    st.session_state.DEFAULTED = True
    st.session_state.INFERRED = False
    st.session_state.UPLOADED = False
    st.session_state.TRAINED = False
    st.session_state.SELECTED = False
    st.session_state.TRACED = False
else:
    st.session_state.DEFAULTED = False

# amdata.var = amdata.var.rename(columns={'status':'control'})
# amdata.var.loc[amdata.var.control=='Down syndrome', 'control'] = 'False'
# amdata.var.loc[amdata.var.control=='healthy', 'control'] = 'True'
# amdata.write_h5ad(amdata_path)

if st.session_state.DEFAULTED:
    amdata_path = 'resources/downsyndrome.h5ad'

    @st.cache_data
    def load_def_amdata():
        amdata = ad.read_h5ad(amdata_path)
        amdata.var.control= amdata.var.control.astype('str')
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


        @st.cache_data
        def load_uploaded(uploaded_file):
            meta= pd.read_csv(uploaded_file, index_col=0)
            if 'control' in meta.columns:
                meta.control = meta.control.astype('str')
            return meta
        

        '### **Upload your methylation data**'
        with st.expander('Expand for more information on the correct format'):
            st.markdown("""
                - All methylation values should be converted to beta values
                - Each row should correspond to a CpG site, each columns to a participant
                - The first column should be an index column with CpG site names
                - The first row should be an index row with participant indexes
                
                <br>
                Correctly formated dataset should look like this after loading:
            """, unsafe_allow_html=True)
            st.image("streamlit/correct_data_format.png")

        uploaded_file1 = st.file_uploader(label='Upload your dataset',label_visibility='collapsed', type=['csv'])
        if uploaded_file1 is not None:
            data = load_uploaded(uploaded_file1)
            'File dimensions: ', data.shape
            # st.dataframe(data.iloc[:5,:5])

        ''
        ''

        '### **Upload your participant metadata**'
        with st.expander('Expand for more information on the correct format'):
            st.markdown("""
                - Each row should correspond to a participant, each column to distinct information
                - The first column should be an index column with participant indexes
                - The file should at least contain columns named exactly 'age' and 'control'
                    - Other columns are allowed and can be used in the downstream analysis
                - The 'age' column shoud contain participant age in years (can be a float)
                - The 'control' should have a value either 'True' standing for healthy individual or 'False' for an individual with a disease
                
                <br>
                Correctly formated dataset should look something like this after loading:
            """, unsafe_allow_html=True)
            st.image("streamlit/correct_metadata_format.png")
        uploaded_file2 = st.file_uploader(label='Upload your metadata',label_visibility='collapsed', type=['csv'])
        if uploaded_file2 is not None:
            meta = load_uploaded(uploaded_file2)
            correct = True

            if 'age' not in meta.columns:
                st.error("The metadata does not have 'age' column!")
                correct = False

            if 'control' not in meta.columns:
                st.error("The metadata does not have 'control' column!")
                correct = False

            meta.control = meta.control.astype('str')
            if (~meta.control.isin(['True', 'False'])).any():
                st.error("The metadata 'control' column has values different than 'True' or 'False'!")
                correct = False

            if correct:
                'File dimensions: ', meta.shape
                # st.dataframe(meta.iloc[np.random.randint(0, meta.shape[0], size=5),:3])
            else:
                meta = None

        ''
        ''

        if data is not None and meta is not None:

            @st.cache_data
            def load_amdata(data, meta):
                t = st.empty()
                t.markdown('Reading the data and metadata... ')
                try:
                    amdata = ad.AnnData(X= data.values,
                            dtype=np.float32,
                            obs= pd.DataFrame(index=data.index),
                            var= meta)
                    
                except Exception as e:
                    st.error("Error loading the data. Make sure it's in the correct format")
                    with st.expander('Expand error log'):
                        e
                t.markdown('Reading the data and metadata ✅')

                t = st.empty()
                t.markdown('Preprocessing... ')
                amdata = preprocess_func.drop_nans(amdata)
                amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
                amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)
                t.markdown('Preprocessing ✅')
                return amdata
            
            amdata=load_amdata(data, meta)
            st.session_state.UPLOADED = True

        ''
        ''


        'Download a list of required CpG sites to filter your dataset (in case your dataset is too big)'
        st.download_button(
            label="⇩ Download CSV (3.6 kB)",
            data=csv,
            file_name='ProbBioAge_cpgs.csv',
            mime='text/csv',
        )

    with tab2:
        if not st.session_state.UPLOADED:
            st.warning('Upload the data and the metadata to run the training.')
        else:
            st.warning("""
            **You do not need to run the training.** Our approach includes a robust batch correction
            that utilizes the information learnt from the Generation Scotland dataset. 
            You can go to the next tab to infer the participant data.
            """)

            from sklearn.feature_selection import r_regression

            @st.cache_data
            def training(_amdata, r2_threshold):

                t = st.empty()
                t.markdown('Computing R2 for every site... ')
                my_bar = st.progress(0)
                r2_array = []
                for i, site_index in enumerate(_amdata.obs.index):
                    my_bar.progress(i/_amdata.n_obs)
                    r2_array.append(r_regression(_amdata[site_index].X.T, _amdata.var.age)[0]**2)
                _amdata.obs['r2'] = r2_array
                my_bar.empty()
                t.markdown('Computing R2 for every site ✅')

                selected, dropped = (_amdata.obs.r2>r2_threshold).value_counts().values
                _amdata = _amdata[_amdata.obs.r2>r2_threshold]
                'Selected ', selected, ' sites based on the r2 threshold (dropped ', dropped, ' sites)'

                t = st.empty()
                t.markdown('Computing MAPs for site parameters... ')
                site_maps = modelling_bio.bio_sites(_amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
                for param in modelling_bio.get_site_params():
                    _amdata.obs[param] = site_maps[param]
                t.markdown('Computing MAPs for site parameters ✅')


                _amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(_amdata, t=90)
                _amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(_amdata)
                _amdata.obs['saturating_der'] = _amdata.obs.abs_der<0.001

                _amdata.obs['saturating'] = _amdata.obs.saturating_std | _amdata.obs.saturating_der

                selected, dropped = _amdata.obs.saturating.value_counts().values
                'Selected ', selected, ' sites based on the saturation threshold (dropped ', dropped, ' sites)'
                # amdata = amdata[~amdata.obs.saturating]
                # 'Selected ', selected, ' sites based on the r2 threshold (dropped ', dropped, ' sites)'

                return _amdata

            r2_threshold = st.slider('R2 threshold for sites that are selected for training', 
                                     min_value=0.0, max_value=1.0, value=0.2)
            if st.button('Run the training'):
                amdata=training(amdata, r2_threshold)

                amdata

                f'Download the learnt site parameters for {amdata.shape[0]} sites'
                st.download_button(
                    label="⇩ Download CSV",
                    data=amdata.obs.to_csv().encode('utf-8'),
                    file_name='ProbBioAge_sites.csv',
                    mime='text/csv',
                )

                st.session_state.TRAINED = True


    with tab3:
        if not st.session_state.UPLOADED:
            st.warning('Upload the data and the metadata to run the batch correction.')
        else:

            @st.cache_data
            def batch_correction(data, meta):

                t = st.empty()
                t.markdown('Reading the data and metadata... ')
                try:
                    amdata = ad.AnnData(X= data.values,
                            dtype=np.float32,
                            obs= pd.DataFrame(index=data.index),
                            var= meta)
                    
                except Exception as e:
                    st.error("Error loading the data. Make sure it's in the correct format")
                    with st.expander('Expand error log'):
                        e
                t.markdown('Reading the data and metadata ✅')

                t = st.empty()
                t.markdown('Preprocessing... ')
                intersection = site_info.index.intersection(amdata.obs.index)
                amdata = amdata[intersection]
                amdata = preprocess_func.drop_nans(amdata)
                amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
                amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)
                amdata.obs[params + ['r2']] = site_info[params + ['r2']]
                t.markdown('Preprocessing ✅')

                t = st.empty()
                t.markdown('Inferring site offsets... ')
                maps = modelling_bio.site_offsets(amdata[:,amdata.var.control=='True'], return_MAP=True, return_trace=False, show_progress=True)['map']
                amdata.obs['offset'] = maps['offset']
                amdata = amdata[amdata.obs.sort_values('offset').index]
                t.markdown('Inferring site offsets ✅')

                t = st.empty()
                t.markdown('Inferring participant accelerations and biases... ')
                amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
                amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset
                ab_maps = modelling_bio.person_model(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
                amdata.var['acc'] = ab_maps['acc']
                amdata.var['bias'] = ab_maps['bias']
                t.markdown('Inferring participant accelerations and biases ✅')
                return amdata

            if st.button('Run the batch correction'):

                col1, col2 = st.columns(2)

                with col1:
                    amdata = batch_correction(data, meta)
                    'Dataset: ', amdata
                with col2:
                    st.pyplot(sns.histplot(amdata.obs.offset).get_figure())

                '---'

                f'Download the acceleration and bias for {amdata.var.shape[0]} participants'
                st.download_button(
                    label="⇩ Download CSV",
                    data=amdata.var[['acc','bias']].to_csv().encode('utf-8'),
                    file_name='ProbBioAge_results.csv',
                    mime='text/csv',
                )

                st.session_state.INFERRED = True




with tab4:
    if st.session_state.INFERRED or st.session_state.DEFAULTED:
    # if amdata is not None:
        'Dataset: ', amdata

        def compute_anova():
            acc_control = amdata[:, amdata.var.control == 'True'].var['acc'].values
            acc_down = amdata[:, amdata.var.control == 'False'].var['acc'].values
            return f_oneway(acc_control, acc_down)
        anova = compute_anova()
        st.write(f'The ANOVA statistic is {round(anova[0],2)} and pvalue is {round(anova[1],2)}')
        

        col1, col2, col3 = st.columns(3)

        columns = amdata.var.columns
        X_axis = col1.selectbox("Choose X axis:", amdata.var.columns, index=amdata.var.columns.to_list().index('acc'))
        Y_axis = col2.selectbox("Choose Y axis:", amdata.var.columns,index=amdata.var.columns.to_list().index('bias'))
        color = col3.selectbox("Choose color column:", amdata.var.columns,index=amdata.var.columns.to_list().index('control'))
        

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
        'DEBUGGING: ', st.session_state.DEBUGGING
        'DEFAULTED: ', st.session_state.DEFAULTED
        'UPLOADED: ', st.session_state.UPLOADED
        'TRAINED: ', st.session_state.TRAINED
        'INFERRED: ', st.session_state.INFERRED
        'SELECTED: ', st.session_state.SELECTED
        'TRACED: ', st.session_state.TRACED

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
