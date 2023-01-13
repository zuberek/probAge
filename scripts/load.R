# Install required packages
library(anndata)

INPUT = "data/wave3_mvals.rds"
# INPUT = "/afs/inf.ed.ac.uk/user/s17/s1768506/disk/methylation/wave1/W1_GS_unrel_mvals.rds"

OUTPUT = "exports/wave3.h5ad"
# OUTPUT = "./exports/wave1.h5ad"


# Takes good 2 minutes to load
print('Loading the Rdata...')
file <- readRDS(INPUT)

print('Converting to anndata...')
file <- anndata::AnnData(file)

print('Saving...')
anndata::write_h5ad(file, OUTPUT)
