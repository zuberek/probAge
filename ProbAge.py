# %load_ext autoreload 
# %autoreload 2

import sys
import streamlit as st

from modules.loading import test_limits, upload_data
from modules import loading
import modules

from src import modelling_bio_beta as modelling
from src import batch_correction as bc

import pandas as pd
import numpy as np
import anndata as ad

# states
if 'SELECTED' not in st.session_state:
    st.session_state.SELECTED = False
if 'TRACED' not in st.session_state:
    st.session_state.TRACED = False

site_info_path = 'resources/ewas_fitted_sites.csv' 

if 'SITE_INFO' not in st.session_state:
    st.session_state.SITE_INFO = pd.read_csv(site_info_path, index_col=0)
if 'PARAMS'  not in st.session_state:
    st.session_state.PARAMS = list(modelling.SITE_PARAMETERS.values())

st.warning('Put the paper here')

data = None
meta = None

data = modules.loading.upload_data()
meta = modules.loading.upload_meta()
amdata = modules.loading.create_anndata(data, meta)


if st.button("Run Inference"):
    
    sites_ref = pd.read_csv('streamlit/wave3_sites.csv', index_col=0)
    # amdata = ad.read_h5ad('resources/downsyndrome.h5ad')

    # Load intersection of sites in new dataset
    params = list(modelling.SITE_PARAMETERS.values())

    intersection = sites_ref.index.intersection(amdata.obs.index)
    amdata.obs[params] = sites_ref[params]

    amdata = amdata[intersection].to_memory()

    t = st.empty()
    t.markdown('Inferring site offsets... ')
    offsets = bc.site_offsets(amdata, show_progress=True)['offset']
    t.markdown('Inferring site offsets ✅')
    
    
    amdata.obs['offset'] = offsets.astype('float64')
    amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
    amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset

    t = st.empty()
    t.markdown('Inferring participants accelerations and biases...  ')
    ab_maps = modelling.person_model(amdata, method='map', progressbar=True)
    t.markdown('Inferring participants accelerations and biases ✅')

    amdata.var['acc'] = ab_maps['acc']
    amdata.var['bias'] = ab_maps['bias']

    st.session_state.DATA = amdata

    # st.switch_page("pages/1_Inference.py")
 

with st.sidebar:
    f'Download demo downsyndrome data'
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