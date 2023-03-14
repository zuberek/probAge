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
if 'DEFAULTED' not in st.session_state:
    st.session_state.DEFAULTED = False
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
    st.session_state.SELECTED = False
    st.session_state.TRACED = False
else:
    st.session_state.DEFAULTED = False

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



if st.session_state.DEFAULTED:
    tab_default = st.tabs(['**Analyse**'])
    tab1, tab2, tab3 = contextlib.nullcontext(), contextlib.nullcontext(), contextlib.nullcontext()
if not st.session_state.DEFAULTED:
    tab_default = contextlib.nullcontext()
    tab1, tab2, tab3 = st.tabs(['**Upload**', '**Infer**', '**Analyse**'])



if not st.session_state.DEFAULTED:
    with tab1:


        @st.cache_data
        def load_uploaded(uploaded_file):
            return pd.read_csv(uploaded_file, index_col=0)
        

        '**Upload your methylation data**'
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
            st.dataframe(data.iloc[:5,:5])

        ''
        ''

        '**Upload your participant metadata**'
        with st.expander('Expand for more information on the correct format'):
            st.markdown("""
                - Each row should correspond to a participant, each column to distinct information
                - The first column should be an index column with participant indexes
                - The file should at least contain columns named exactly 'age' and 'status'
                    - Other columns are allowed and can be used in the downstream analysis
                - The 'age' column shoud contain participant age in years (can be a float)
                - The 'status' should have a value either 'healthy' or 'disease'
                
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

            if 'status' not in meta.columns:
                st.error("The metadata does not have 'status' column!")
                correct = False

            if (~meta.status.isin(['disease', 'healthy'])).any():
                st.error("The metadata 'status' column has values different than 'disease' or 'healthy'!")
                correct = False

            if correct:
                'File dimensions: ', meta.shape
                st.dataframe(meta.iloc[:5,:3])
            else:
                meta = None

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
        if data is None or meta is None:
            st.warning('Upload the data and the metadata to run the batch correction.')
        else:


            if st.button('Run the batch correction'):
                st.session_state.INFERRED = True

            if st.session_state.INFERRED:
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
                    maps = modelling_bio.site_offsets(amdata[:,amdata.var.status=='healthy'], return_MAP=True, return_trace=False, show_progress=True)['map']
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



with tab3:

    if amdata is not None:
        'Dataset: ', amdata

        @st.cache_data
        def compute_anova():
            acc_control = amdata[:, amdata.var.status == 'healthy'].var['acc'].values
            acc_down = amdata[:, amdata.var.status == 'disease'].var['acc'].values
            return f_oneway(acc_control, acc_down)
        anova = compute_anova()
        st.write(f'The ANOVA statistic is {round(anova[0],2)} and pvalue is {round(anova[1],2)}')
        

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
                        return modelling_bio.person_model(amdata=amdata[:, person_index],
                            return_trace=True, return_MAP=False, show_progress=True)['trace']
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
        'DEBUGGING: ', st.session_state.DEBUGGING
        'DEFAULTED: ', st.session_state.DEFAULTED
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
