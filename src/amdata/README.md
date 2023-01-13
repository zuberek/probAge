# Annotated Methylation Data

This is building on the Annotated Data (anndata) package:
- [Anndata basic docs](https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.html)
- [Anndata more in depth](https://falexwolf.de/blog/2017-12-23-anndata-indexing-views-HDF5-backing/)

The main additions so far are:
- **Naming**: I've added aliases for the observations to be cpg sites and variables to be data samples. You can instead of `amdata.obs` say `amdata.sites` and instead of `amdata.var` say `amdata.samples`. Similarly you can do `amdata.n_samples` and `amdata.n_sites` etc.
- **Plotting**: You can quickly plot a single CpG site with a quite flexible `amdata.plot.site(site)`. Look at the function for more information
- **Binning**: You can quickly bin the sites along the age in various ways saying `amdata.bin()`

Basic loading looks like this:
```
hannum = amdata.AnnMethylData(HANNUM_PATH)

# If the file is big, load it lazily passing the backed parameter
# Look at the read_h5ad docs of the anndata for more info
hannum = amdata.AnnMethylData(HANNUM_PATH, backed='r+')

```