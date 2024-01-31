import streamlit as st

import pandas as pd
import numpy as np
from scipy.stats import beta

import matplotlib.pyplot as plt

st.title('Interactive Exploration of Methylation Dynamics.')

with st.sidebar:

    st.markdown('# Model parameters')
    p = st.slider('Initial methylation level,\t ' + r'$p$', 0., 1., 0.1, 0.01) 
    eta_0 = st.slider('Terminal methylation level,\t' + r'$\eta_0$', 0., 1., 0.9, 0.01) 
    omega = st.slider('Speed,\t' + r'$\omega$', 0., 0.05, 0.02, 0.001)
    var_init = st.slider('Initial variance', 0.0001, 100. , 10., 10.)
    N = st.slider('System size', 10., 1000., 100., 100.)
    acc = st.slider('Acceleration', -1., 1., 0., 0.1)
    bias = st.slider('Bias', -0.1, 0.1, 0., 0.01)

col1, col2 = st.columns([4, 1])

show_dots = col2.checkbox(
    'Show dots', value=True)

show_stats = col2.checkbox(
    'Show mean and 95% CI', value=True)

fig, ax = plt.subplots()

t = np.linspace(0,100,1_001)

# update acc and bias
omega = np.exp(acc)*omega

# reparametrization
eta_1 = 1 - eta_0

# model mean and variance
var_term_0 = eta_0*eta_1
var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)

mean = eta_0 + np.exp(-omega*t)*((p-1)*eta_0 + p*eta_1)

#update bias
mean = mean + bias

variance = (var_term_0/N 
        + np.exp(-omega*t)*(var_term_1-var_term_0)/N 
        + np.exp(-2*omega*t)*(var_init/np.power(N,2) - var_term_1/N)
    )

k = (mean*(1-mean)/variance)-1
a = mean*k
b = (1-mean)*k

if show_stats is True:
    conf_int = np.array(beta.interval(0.95, a, b))
    low_conf = conf_int[0]
    upper_conf = conf_int[1]
    plt.plot(t, mean, 'r', label='Mean')
    plt.plot(t, low_conf, 'c--', label='95% CI')
    plt.plot(t, upper_conf, 'c--', )
    plt.ylim(0,1)
    plt.legend()

if show_dots is True:

    samples = beta.rvs(a[:, None], b[:, None], size=(1_001, 100))
    samples = samples.flatten()
    new_t = np.repeat(t,100)

    plt.scatter(new_t, samples, alpha=0.05, s=0.2)
    plt.ylim(0,1)


col1.pyplot(fig)