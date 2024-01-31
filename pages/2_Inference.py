import streamlit as st
import plotly.express as px
import seaborn as sns
import numpy as np
import arviz as az

from src import modelling_bio_beta as modelling

from streamlit_plotly_events import plotly_events

if 'DATA' not in st.session_state:
    st.warning('Upload data and metadata to run model inference.')
else:
    amdata = st.session_state.DATA

    f'Download the acceleration and bias for {amdata.var.shape[0]} participants'
    st.download_button(
        label="⇩ Download CSV",
        data=amdata.var.to_csv().encode('utf-8'),
        file_name='ProbAge_results.csv',
        mime='text/csv',
    )


    fig = px.scatter(data_frame=amdata.var, x='acc', y='bias', color='status', 
                marginal_x='box', marginal_y='box',hover_name=amdata.var.index)

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ))

    selected_points = plotly_events(fig)

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
                    return modelling.person_model(amdata=amdata[:, person_index],
                        method='nuts', progressbar=True)
                trace=compute_trace(person_index)

                # trace.posterior.part.values

                st.pyplot(az.plot_pair(trace,kind='kde').get_figure())

