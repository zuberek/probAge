# %%
# %load_ext autoreload
# %autoreload 2
# import sys
# sys.path.append("../..")   # fix to import modules from root
import anndata as ad
from anndata._core.file_backing import AnnDataFileManager 
import pandas as pd
import numpy as np
import seaborn as sns

from src.utils import plot
from tqdm import tqdm

# from src.amdata import ammodel
from src.amdata import amplot


# %%
class MethylDataFrame(pd.DataFrame):

    def histogram(self, x, y=None, hue=None, cbar=None, hue_order=None, bins=50, ax=None, **kwargs):

        if cbar is None:
            cbar = True if hue is None else False # don't show with hue

        if ax is None:
            title = f'Histogram of {x}'
            if y is not None: title += f' vs {y}'
            if hue is not None: title += f' and hue {hue}'

            ax = plot.row(title)

        if hue=='modality': hue_order=[4,3,2,1]

        sns.histplot(self, x=x, y=y, hue=hue, palette='tab10', cbar=cbar,
                    hue_order=hue_order, bins=bins, ax=ax, **kwargs)

    def scatter(self, x, y, hue=None, hue_order=None, ax=None, **kwargs):
        if ax is None:
            ax = plot.row(f'Scatterplot of {x} vs {y}')

        if hue=='modality': hue_order=[4,3,2,1]

        sns.scatterplot(data=self, x=x, y=y, hue=hue, palette='tab10',
                    hue_order=hue_order, ax=ax, **kwargs)
        


class AnnMethylData(ad.AnnData):

    do_overwrite = False
    path = ''
    CHUNKIFY=160

    def __init__(self, X=None, oidx=None, vidx=None, filename=None, backed=None, asview=False, do_overwrite=False):
        self.do_overwrite = do_overwrite
        if isinstance(X, str):
            self.path = X
            X = ad.read_h5ad(X, backed=backed)
        #     if isinstance(X, AnnData)
            # X = ad.read_h5ad(X, backed=backed)
            # super().__init__(ad.read_h5ad(X, backed=backed))
            # if isinstance(backed, str):
                # super().filename(X)

        if filename is not None:
            self.path = filename
            X = ad.read_h5ad(filename, backed=backed)

        super().__init__(X=X, oidx=oidx, vidx=vidx, asview=asview)

        if isinstance(backed, str):
            self.file = AnnDataFileManager(self, self.path, backed)
            self.filename = self.path

        # give alternative names 
        self.sites = self.obs
        self.samples = self.var

        self.n_sites = self.n_obs
        self.n_samples = self.n_vars
        # self.sites = MethylDataFrame(self.obs)
        # self.samples = MethylDataFrame(self.var)

        # load additional modules
        # self.model = ammodel.AnnMethylModeller(self)
        self.plot = amplot.AnnMethylPlotter(self)

    def __getitem__(self, index) -> "AnnMethylData":
        """Returns a sliced view of the object."""
        oidx, vidx = self._normalize_indices(index)
        return AnnMethylData(X=self, oidx=oidx, vidx=vidx, asview=True, filename=self.filename, backed=self.file._filemode)

    def estimate_chunksize():
        total_memory = 40*1024 # 40 GB
        part_to_use = 0.3
        single_row = self[0].X.__sizeof__()/1e6
        single_row = 0.0179
        chunk_memory = total_memory*part_to_use
        chunksize = chunk_memory/single_row
        return chunksize

    def chunkify(self, chunksize=10_000):

        n_chunks = self.n_sites//chunksize
        self.n_chunks = n_chunks

        # self.sites['chunk'] = pd.Series()
        # chunk_position = np.nonzero(self.sites.columns=='chunk')[0][0]
        # for chunk_idx in tqdm(range(n_chunks+1)):
        #     start = chunk_idx*chunksize
        #     end = start+chunksize
        #     end = end if end <= self.n_sites else self.n_sites
        #     assert start<=end
        #     self.sites.iloc[range(start,end), chunk_position] = chunk_idx

        chunks = []
        for chunk_idx in tqdm(range(n_chunks+1)):
            start = chunk_idx*chunksize
            end = start+chunksize
            end = end if end <= self.n_sites else self.n_sites
            assert start<=end
            chunks.append(self[range(start,end)].sites.index)

        return chunks

    def ranges(self, chunksize=10000):

        n_chunks = self.n_sites//chunksize
        self.n_chunks = n_chunks

        ranges = []
        for chunk_idx in tqdm(range(n_chunks+1)):
            start = chunk_idx*chunksize
            end = start+chunksize
            end = end if end <= self.n_sites else self.n_sites
            assert start<=end
            ranges.append(range(start,end))

        return ranges

    def parallelize(self, chunk, func, chunksize=None):
        if chunksize is None:
            chunksize = self.n_obs//self.CHUNKIFY
        with Pool() as p:
            result = np.array(list(tqdm(
                p.imap(lambda site_names: func(self[site_names]), self.obs.index, chunksize=chunksize), 
                    total=self.n_obs)))
        return result



    def bin(self, n_bins=10, equal_bins=False):

        if equal_bins:
            vars_copy = self.var.sort_values(by='age').reset_index()

            bins = vars_copy.groupby(
                vars_copy.index//(self.n_vars/n_bins)).apply(
                lambda x: pd.Series(    # for each group make a row
                    [len(x), int(x['age'].mean()), x['age'].min(), x['age'].max()],  
                    index=['count','center','left','right']))

        else:
            hist, bin_edges = np.histogram(self.var.age, bins=n_bins)
            bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
            bins = pd.DataFrame([hist, bin_centers, bin_edges[:-1], bin_edges[1:]], index=['count','center', 'left','right']).T
        
        self.uns['bins'] = bins
        self.var['bin'] = pd.Series()
        # self.var['bin_age'] = pd.Series()
        # for bin_idx in range(len(bin_edges)-1):
        #     bin_mask = (self.var.age>=bin_edges[bin_idx]) & (self.var.age<=bin_edges[bin_idx+1])
        #     # bin_age = self.var[bin_mask].age.mean()
        #     self.var.loc[bin_mask, 'bin'] = bin_idx
        #     # self.var.loc[bin_mask, 'bin_age'] = bin_age
        # self.var['bin_age'] = pd.Series()

        for bin_idx, bin in bins.iterrows():
            bin_mask = (self.var.age>=bin.left) & (self.var.age<=bin.right)
            self.var.loc[bin_mask, 'bin'] = bin_idx

    def bin_mean(self):
        bin_results = np.empty(shape=(self.n_obs, len(self.uns['bins'])))
        for bin_idx in tqdm(range(len(self.uns['bins']))):
            bin_results[:, bin_idx] = self[:, self.var.bin==bin_idx].X.mean(axis=1)
        # self.uns['variances'] = pd.DataFrame(bin_results, index=self.obs.index)
        return bin_results

    def bin_variance_site(self, site_name):
        if 'bins' not in self.uns.keys():
            self.bin(equal_bins=True)
        site_meta, site_obs, site_values = self.site(site_name)
        
        y_means = [site_values[self.samples[self.samples.bin==bin_idx].pos].mean() for bin_idx in self.uns['bins'].index]
        y_vars = [site_values[self.samples[self.samples.bin==bin_idx].pos].var() for bin_idx in self.uns['bins'].index]

        return np.array(y_means), np.array(y_vars)


    def bin_variance(self):
        sites_variances = np.empty(shape=(self.n_obs, len(self.uns['bins'])))
        for bin_idx in tqdm(range(len(self.uns['bins']))):
            sites_variances[:, bin_idx] = self[:, self.var.bin==bin_idx].X.var(axis=1)
        # self.uns['variances'] = pd.DataFrame(sites_variances, index=self.obs.index)
        return sites_variances


    def bin_normalize(pdfs, n_bins=10, equal_bins=True):
        selected.bin(n_bins, equal_bins=equal_bins)

        for bin_idx in range(n_bins):
            bin_sites_pdf = pdfs[selected.var.bin==bin_idx]
            if len(bin_sites_pdf)==0:
                continue
            scaled = (bin_sites_pdf - 0) / (np.max(bin_sites_pdf) - 0)
            pdfs[selected.var.bin==bin_idx] = scaled

        return pdfs
    # pdfs_all=normalize(pdfs_all.values.T)



    def site(self, site_name, sample=slice(None), var2obs=False):
        '''
        Return basic info for particular site
        '''
         
        if var2obs:
            site_meta = self.var.loc[site_name].squeeze()
            site_obs = self.obs.squeeze()
            site_values = self[:, site_name].X.flatten()
        else:
            site_meta = self[site_name, sample].obs.squeeze()
            site_obs = self[site_name, sample].var.squeeze()
            site_values = self[site_name, sample].X[0]
        return site_meta, site_obs, site_values


    def overwrite(self):
        self.do_overwrite = True

    def save(self):
        if self.do_overwrite:
            self.write_h5ad(self.path)
        else:
            print('The AnnMethylData object was not initiated to overwrite!')

    # def save_selected(self):
    #     if self.do_overwrite:
    #         self.write_h5ad(f"{self.path}/hannum_data_selected.h5ad")
    #     else:
    #         print('The AnnMethylData object was not initiated to overwrite!')

    # def load_selected(self):
    #     return AnnMethylData(ad.read_h5ad(f"{self.path}/hannum_data_selected.h5ad"))

