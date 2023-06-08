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
pio.templates.default = "plotly"

import contextlib 

import src.modelling_bio  as modelling_bio
# from src.modelling_bio import person_model

site_info_path = 'resources/wave3_acc_sites.csv' 


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

def drop_nans(amdata):
    nans = np.isnan(amdata.X).sum(axis=1).astype('bool')
    print(f'There were {nans.sum()} NaNs dropped')
    # Use the ~ character to flip the boolean
    return amdata[~nans]


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


# %% 
# LOAD DATA
amdata = None
person_index = None
selected_points = []

@st.cache_data
def load_site_info():
    site_info = pd.read_csv(site_info_path, index_col=0)
    params = get_site_params()
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
'# ProbBioAge'


use_default = st.checkbox('Use the default downsyndrome dataset')

if use_default:
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
if not use_default:
    tab_default = contextlib.nullcontext()
    tab1, tab2, tab3 = st.tabs(['**Upload**', '**Correct**', '**Analyse**'])



if not use_default:
    with tab1:
        
        data = None
        meta = None

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
            st.warning('Upload the data and the metadata to run the batch correction')
        else:

            if st.button('Run the batch correction'):

                t = st.empty()
                t.markdown('Reading the data and metadata... ')
                @st.cache_data
                def load_amdata(data, meta):
                    return ad.AnnData(X= data.values,
                            dtype=np.float32,
                            obs= pd.DataFrame(index=data.index),
                            var= meta)
                try:
                    amdata = load_amdata(data, meta)
                    t.markdown('Reading the data and metadata ✅')
                except Exception as e:
                    st.error("Error loading the data. Make sure it's in the correct format")
                    with st.expander('Expand error log'):
                        e
                
                t = st.empty()
                t.markdown('Preprocessing... ')
                intersection = site_info.index.intersection(amdata.obs.index)
                amdata = drop_nans(amdata)
                amdata.X = np.where(amdata.X == 0, 0.00001, amdata.X)
                amdata.X = np.where(amdata.X == 1, 0.99999, amdata.X)
                amdata.obs[params + ['r2']] = site_info[params + ['r2']]
                t.markdown('Preprocessing ✅')


                t = st.empty()
                t.markdown('Inferring site offsets... ')
                maps = modelling_bio.site_offsets(amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
                amdata.obs['offset'] = maps['offset']
                sns.histplot(amdata.obs.offset)
                amdata = amdata[amdata.obs.sort_values('offset').index]


with tab3:
    'Dataset: ', amdata

    if amdata is not None:

        
        # @st.cache_data
        # def compute_anova():
        #     acc_control = amdata[:, amdata.var.status == 'healthy'].var['acc'].values
        #     acc_down = amdata[:, amdata.var.status == 'disease'].var['acc'].values
        #     return f_oneway(acc_control, acc_down)
        # anova = compute_anova()
        # st.write(f'The ANOVA statistic is {round(anova[0],2)} and pvalue is {round(anova[1],2)}')
        

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
        st.warning('Either use the default downsyndrome dataset, or upload the data and the metadata and run the batch correction to analyse the dataset')


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
