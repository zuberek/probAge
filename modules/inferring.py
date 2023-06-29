import streamlit as st
import src.modelling_bio  as modelling_bio
from sklearn.feature_selection import r_regression
import seaborn as sns

# @st.cache_data(experimental_allow_widgets=True)
# def infer(status, _amdata):

#     if status=='Infer using the corrected GS model':
#         if st.button('Run the batch correction'):
#             _amdata = batch_correction(_amdata)

#             return _amdata

        
#     if status=='Infer using the trained model':
#         _amdata = model_retraining(_amdata)

#         return _amdata
#     st.write(st.session_state.INFERRED)

@st.cache_data
def batch_correction(status, _amdata):

    col1, col2 = st.columns(2)

    with col1:
        t = st.empty()
        t.markdown('Preprocessing... ')
        intersection = st.session_state.SITE_INFO.index.intersection(_amdata.obs.index)
        _amdata = _amdata[intersection]
        _amdata.obs[st.session_state.PARAMS + ['r2']] = st.session_state.SITE_INFO[st.session_state.PARAMS + ['r2']]
        t.markdown('Preprocessing ✅')

        t = st.empty()
        t.markdown('Inferring site offsets... ')
        maps = modelling_bio.site_offsets(_amdata[:,_amdata.var.status=='control'], return_MAP=True, return_trace=False, show_progress=True)['map']
        _amdata.obs['offset'] = maps['offset']
        _amdata = _amdata[_amdata.obs.sort_values('offset').index]
        t.markdown('Inferring site offsets ✅')

        t = st.empty()
        t.markdown('Inferring participant accelerations and biases... ')
        _amdata.obs.eta_0 = _amdata.obs.eta_0 + _amdata.obs.offset
        _amdata.obs.meth_init  = _amdata.obs.meth_init + _amdata.obs.offset
        ab_maps = modelling_bio.person_model(_amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
        _amdata.var['acc'] = ab_maps['acc']
        _amdata.var['bias'] = ab_maps['bias']
        t.markdown('Inferring participant accelerations and biases ✅')

    
    with col2:
        st.pyplot(sns.histplot(_amdata.obs.offset).get_figure())

    return _amdata






@st.cache_data(experimental_allow_widgets=True)
def model_retraining(status, _amdata):

    st.write('### Model training')

    # st.warning("""
    # You can also upload site parameters if you have already trained the model.
    # """)

    # uploaded_file3 = st.file_uploader(label='Upload your site parameters',label_visibility='collapsed', type=['csv'])
    # if uploaded_file3 is not None:
    #     site_meta = load_uploaded(uploaded_file3)
    #     'File dimensions: ', site_meta.shape
    #     amdata = amdata[site_meta.index]
    #     amdata.obs = site_meta
    #     st.session_state.TRAINED = True

    r2_threshold = st.slider('R2 threshold for sites that are selected for training', 
            min_value=0.0, max_value=1.0, value=0.2)

    if st.button('Run the inference'):
        
        @st.cache_data
        def computing_r2(_amdata):

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
            return _amdata

        @st.cache_data
        def retraining(_amdata):
            selected, dropped = (_amdata.obs.r2>r2_threshold).value_counts().values
            _amdata = _amdata[_amdata.obs.r2>r2_threshold]
            'Selected ', selected, ' sites based on the r2 threshold (dropped ', dropped, ' sites)'

            t = st.empty()
            t.markdown('Computing MAPs for site parameters... ')
            site_maps = modelling_bio.bio_sites(_amdata[:, _amdata.var.status=='control'], return_MAP=True, return_trace=False, show_progress=True)['map']
            for param in modelling_bio.get_site_params():
                _amdata.obs[param] = site_maps[param]
            t.markdown('Computing MAPs for site parameters ✅')

            _amdata.obs['abs_der'] = modelling_bio.mean_abs_derivative_at_point(_amdata, t=90)
            _amdata.obs['saturating_std'] = modelling_bio.is_saturating_vect(_amdata)
            _amdata.obs['saturating_der'] = _amdata.obs.abs_der<0.001

            _amdata.obs['saturating'] = _amdata.obs.saturating_std | _amdata.obs.saturating_der
            return _amdata

        @st.cache_data
        def inferring(_amdata):

            selected, dropped = _amdata.obs.saturating.value_counts().values
            st.write('There are ', selected, ' saturating sites sites (', dropped, ' not saturating sites)')
            _amdata = _amdata[~_amdata.obs.saturating]

            if _amdata.n_obs>250:
                st.write('Selecting best 250 r2 CpG sites')
                _amdata=_amdata[_amdata.obs.sort_values('r2').tail(250).index]

            t = st.empty()
            t.markdown('Inferring participant accelerations and biases... ')
            ab_maps = modelling_bio.person_model(_amdata, return_MAP=True, return_trace=False, show_progress=True)['map']
            _amdata.var['acc'] = ab_maps['acc']
            _amdata.var['bias'] = ab_maps['bias']
            t.markdown('Inferring participant accelerations and biases ✅')
            return _amdata

        col1, col2 = st.columns(2)

        with col1:
            _amdata = computing_r2(_amdata)
        with col2:
            ax=sns.histplot(_amdata.obs.r2)
            ax.axvline(x=r2_threshold, ymin=0, ymax=1)
            st.pyplot(ax.get_figure())

        with col1:
            _amdata = retraining(_amdata)
            site_params = _amdata.obs
            _amdata = inferring(_amdata)                  

        st.session_state.INFERRED = True

        return _amdata