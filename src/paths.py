DATASET = 'ewas'
DATASET = 'Nelly'

# Raw
DATA_RAW = '../data/Patients_betas_reduced_ready-40.csv'
DATA_PATIENTS = '../data/Annotation_paediatric_40.csv'

# Processed
# DATA_PROCESSED = '../exports/wave1_meta.h5ad'
# DATA_PROCESSED = '../exports/Nelly_raw.h5ad'
DATA_PROCESSED = f'../exports/{DATASET}_meta.h5ad'

# Fitted
# DATA_FITTED = '../exports/Nelly_fitted.h5ad'
# DATA_FITTED = '../exports/wave1_fitted.h5ad'
# DATA_FITTED = '../exports/wave3_linear.h5ad'
DATA_FITTED = f'../exports/{DATASET}_fitted.h5ad'

# Other
DATA_REFERENCE = '../exports/ewas_fitted.h5ad'
# DATA_REFERENCE = '../exports/wave3_MAP_acc.h5ad'
