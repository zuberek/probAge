# %% ########################
### LOADING

%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from scipy.stats import linregress


amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/tissue_horvath_filtered.h5ad')
amdata = amdata.T
tissues = list(amdata.var.tissue.unique())


# %%

all_sites = pd.DataFrame(index=tissues)
# site_index = amdata.obs.index[1]
for site_index in tqdm(amdata.obs.index[:100]):
    site_slopes = []

    for tissue in tissues:
        tissue_data = amdata[:, amdata.var.tissue==tissue].copy()

        res=linregress(tissue_data.var.age, tissue_data[site_index].X.flatten())
        site_slopes.append(res[0])

    all_sites[site_index] = site_slopes

# %%

sns.heatmap(data=all_sites)