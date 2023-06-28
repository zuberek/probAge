import streamlit as st
import pandas as pd
import numpy as np
import anndata as ad

import src.preprocess_func  as preprocess_func

def test_limits():
    st.write('Limit reached!')


@st.cache_data
def load_uploaded(uploaded_file):
    data= pd.read_csv(uploaded_file, index_col=0)
    if 'status' in data.columns:
        data.status = data.status.astype('str')
    return data

def upload_data():
    
    st.write('### **Upload your methylation data**')
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
        st.write('File dimensions: ', data.shape)
        return data

def upload_meta():

    st.write('### **Upload your participant metadata**')
    with st.expander('Expand for more information on the correct format'):
        st.markdown("""
            - Each row should correspond to a participant, each column to distinct information
            - The first column should be an index column with participant indexes
            - The file should at least contain columns named exactly 'age' and 'status'
                - Other columns are allowed and can be used in the downstream analysis
            - The 'age' column shoud contain participant age in years (can be a float)
            - The 'status' should have a string value of either 'control' standing for healthy individual or 'test' for an individual with a disease
            
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

        if (~meta.status.isin(['control', 'test'])).any():
            st.error("The metadata 'status' column has values different than 'control' or 'test'!")
            correct = False

        if correct:
            st.write('File dimensions: ', meta.shape)
            return meta
            # st.dataframe(meta.sample(5).iloc[:,:3])
        else:
            meta = None

def create_anndata(data, meta):
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

        return amdata