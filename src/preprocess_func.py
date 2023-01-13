
def convert_mvalues_to_betavalues(amdata):
    2**amdata.X/(2**amdata.X + 1)

def merge_meta(amdata, meta):
    amdata.obs['pos'] = range(amdata.n_obs) # to have numerical index stored as well
    amdata.var['pos'] = range(amdata.n_vars)
    amdata.var[['name', 'age', 'sex']] = meta[['Sample_Name', 'age', 'sex']]
 