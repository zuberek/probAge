# %%

%load_ext autoreload 
%autoreload 2
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

from src import modelling
import arviz as az

import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

N_SITES =  10

amdata = amdata.AnnMethylData('../exports/wave3_linear.h5ad')
amdata = amdata[amdata.obs.sort_values('r2', ascending=False).index[:N_SITES]]

# sns.scatterplot(x= amdata.var.age, y=amdata['cg07547549'].X.flatten())

with Pool(15, maxtasksperchild=1) as p:
    results = list(tqdm(
            iterable= p.imap(
                func=modelling.fit_and_compare_2,
                iterable=amdata, 
                ), 
            total=amdata.n_obs))

#modelling.comparison_postprocess(results, amdata)

# %%
comparison = modelling.comparison_postprocess(results, amdata)

# %%
results[9][1]

# %%
#amdata.write_h5ad('../exports/wave3_linear.h5ad')

def chemical_plot (nu_0, nu_1, p, var_init, N, site=None):
  omega = nu_0 + nu_1
  eta_0 = nu_0/omega
  eta_1 = nu_1/omega

  t= np.linspace(0,200, 1_000)
  mean = eta_0 + np.exp(-omega*t)*((p-1)*eta_0 + p*eta_1)

  var_term_0 = eta_0*eta_1
  var_term_1 = (1-p)*np.power(eta_0,2) + p*np.power(eta_1,2)


  var = (var_term_0/N 
          + np.exp(-omega*t)*(var_term_1-var_term_0)/N 
          + np.exp(-2*omega*t)*(var_init/np.power(N,2) - var_term_1/N)
        )

  sns.lineplot(x=t, y=mean, color='orange')
  sns.lineplot(x=t, y=mean+2*np.sqrt(var), color='orange')
  sns.lineplot(x=t, y=mean-2*np.sqrt(var), color='orange')

  if site != None:
    sns.scatterplot(x=amdata[site].participants.age, y=amdata[site].X.flatten())
  # sns.lineplot(x=t, y=mean_linear, color='orange')
  # sns.lineplot(x=t, y=mean_linear+2*np.sqrt(var_linear), color='orange')
  # sns.lineplot(x=t, y=mean_linear-2*np.sqrt(var_linear), color='orange')


# params = [max_p['nu_0'], max_p['nu_1'], max_p['var_init'], max_p['init_meth'],  max_p['system_size']]


results[0]
chemical_plot(*list(var_mean), site=1)
plt.ylim(0,1)