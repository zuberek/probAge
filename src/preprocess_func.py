import numpy as np

def drop_nans(amdata):
    nans = np.isnan(amdata.X).sum(axis=1).astype('bool')
    print(f'There were {nans.sum()} NaNs dropped')
    # Use the ~ character to flip the boolean
    return amdata[~nans]

def convert_mvalues_to_betavalues(amdata):
    return 2**amdata.X/(2**amdata.X + 1)

def merge_meta(amdata, meta, pheno, survival, clock_results, clock_sites): 

    # Add basic metadata
    ####################
    amdata.var['pos'] = range(amdata.n_vars)
    amdata.obs['pos'] = range(amdata.n_obs) # to have numerical index stored as well
    amdata.var[['name', 'age', 'sex']] = meta[['Sample_Name', 'age', 'sex']]

    # Add phenotype data and survival data
    ####################
    columns = ['units', 'usual', 'ever_smoke', 'pack_years',
       'bmi', 'body_fat', 'simd', 'edu_years']
    amdata.var[columns] = pheno[columns]
    amdata.var[['Event', 'tte']] = survival[['Event', 'tte']]

    # Create weighted_smoke phenotype
    # Normalize pack_years data
    amdata.var['norm_pack_years'] = np.log(1+amdata.var.pack_years)

    # Combine ever_smoke with pack_years
    amdata.obs['weighted_smoke'] = amdata.obs['norm_pack_years']/np.exp(amdata.obs['ever_smoke'])

    # Add accelerations given to genscot participant by other clocks
    ####################
    # rename columns to some manageable naming 
    clock_results = clock_results.rename(columns={
       'AgeAccelerationResidualHannum': 'Hannum',
       'EEAA': 'Horvath',
       'AgeAccelGrim': 'Grim',
       'AgeAccelPheno': 'Pheno',
       })
    clock_columns = ['Hannum','Horvath','Grim','Pheno']
    amdata.var[clock_columns] = clock_results[clock_columns]

    # Mark sites that were already used in different clocks
    ####################
    # remove sites that are used in clocks but not in generation scotland
    clock_sites = clock_sites.loc[clock_sites.index.intersection(amdata.obs.index)]

    for group_name, group_data in clock_sites.groupby('Clock'):
        amdata.obs[group_name] = False
        amdata.obs.loc[group_data.index, group_name] = True




 