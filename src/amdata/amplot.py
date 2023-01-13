from src.utils import plot

import numpy as np
import seaborn as sns

class AnnMethylPlotter():

    def __init__(self, amdata):
        self.amdata = amdata

    def lemon(self):
        print(self.amdata)

    def site(self, site_name=None, ages=None, site_values=None, sample=None, linefit=False, var2obs=False, highlight=None, hue=None, observation=True, ax=None, zoom=False, pred=None, **kwargs):
        if site_name is None:
            if (site_values is None) or (ages is None):
                print('You need to pass either the site name or the x/y values!')

        # sample_idxs = slice(None) if sample is None else self.var.sample(sample).index

        if sample is not None:
            sample_idxs = self.amdata.var.sample(sample).index
            if hue is not None:
                hue = hue[sample_idxs] 
        else:
            sample_idxs = slice(None)

        if site_name is not None:
            site_meta, site_obs, site_values = self.amdata.site(site_name, sample=sample_idxs, var2obs=var2obs)
            ages = site_obs.age


        # site_meta = self.amdata[site_name].obs.squeeze()
        # site_values = self.amdata[site_name].X[0]
        # ages = self.amdata.var.age.values

        if ax is None:
            ax = plot.row([f'Site {site_name}'])
        elif isinstance(ax, str):
            ax = plot.row([ax])

        if not zoom:
            ax.set_xlim([15, 100])
            ax.set_ylim([0, 1])

        sns.scatterplot(x=ages, y=site_values, hue=hue, ax=ax, **kwargs)
        ax.set_xlabel('Age')
        ax.set_ylabel('CpG value')

        if linefit:
            x_start_end = np.array([ages.min(), ages.max()])
            y_start_end = site_meta.slope*x_start_end + site_meta.intercept
            sns.lineplot(x=x_start_end, y=y_start_end, color='blue', ax=ax, label='linefit') 

            if 'slope_var' in self.amdata.obs.columns:
                y_means = site_meta.slope*ages + site_meta.intercept
                y_stds = np.sqrt(site_meta.slope_var*ages)
                sns.lineplot(x=ages, y=y_means+y_stds*2, color='blue', ax=ax)
                sns.lineplot(x=ages, y=y_means-y_stds*2, color='blue', ax=ax)

        if highlight is not None:
            if isinstance(highlight, list):
                sns.scatterplot(x=ages[highlight], y=site_values[highlight], 
                                s=100, color="red", label=f'Population', ax=ax)  
            else:
                sns.scatterplot(x=[ages[highlight]], y=[site_values[highlight]], 
                                s=100, color="red", label=f'Person {highlight}', ax=ax)

        if observation :
            # if (not 'bin_variances' in self.amdata.uns.keys()) and (not 'bin_means' in self.amdata.uns.keys()):
            y_means, y_vars = self.amdata.bin_variance_site(site_name)
                # site_meta, site_obs, site_values = self.amdata.site(site_name, var2obs=var2obs)
                # y_means = [site_values[site_obs[site_obs.bin==bin_idx].pos].mean() for bin_idx in self.amdata.uns['bins'].index]
                # y_vars = [site_values[site_obs[site_obs.bin==bin_idx].pos].var() for bin_idx in self.amdata.uns['bins'].index]
                
                # print('You need to complete the variance analysis to plot observations...')
                # print('Computing bin variances...')
                # self.amdata.uns['bin_variances'] = self.amdata.bin_variance()
                # print('Computing bin means...')
                # self.amdata.uns['bin_means'] = self.amdata.bin_mean()
            # else:
            #     y_means = self.amdata.uns['bin_means'][site_meta.pos]
            #     y_vars = self.amdata.uns['bin_variances'][site_meta.pos]

            y_stds = np.sqrt(y_vars)
            bin_centers = self.amdata.uns['bins'].center

            sns.lineplot(x=bin_centers, y=y_means+y_stds*2, color='k', label='observation', ax=ax)
            sns.lineplot(x=bin_centers, y=y_means-y_stds*2, color='k', ax=ax)
            sns.lineplot(x=bin_centers, y=y_means, color='k', ax=ax)

        if pred is not None:
            y_means, y_stds = pred
            y_means = y_means.loc[sample_idxs]
            y_stds = y_stds.loc[sample_idxs]
            sns.lineplot(x=[ages.sort_values().index[0],ages.sort_values().index[-1]], y=[y_means.loc[ages.sort_values().index[0]],y_means.loc[ages.sort_values().index[-1]]], color='red', ax=ax, label='prediction')
            sns.lineplot(x=ages, y=y_means+y_stds*2, color='red', ax=ax)
            sns.lineplot(x=ages, y=y_means-y_stds*2, color='red', ax=ax)
# site_variances = selected.uns['variances'].loc[site_name]
# y_mean = (site_meta.slope*selected.uns['bins'].center + site_meta.intercept)
# y_std = np.sqrt(site_variances)
# sns.lineplot(x=ages, y=y_mean+y_std*2, ax=ax)
# sns.lineplot(x=ages, y=y_mean-y_std*2, ax=ax)


        return ax

    def sites(self, site_names, col=False, title=None, zoom=False, highlight=None, axs=None):
        if axs is None:
            if col:
                axs = plot.col(site_names, title)
            else:
                axs = plot.row(site_names, title)


        for i, site_name in enumerate(site_names):
            self.site(site_name, ax=axs[i], zoom=zoom, highlight=highlight)