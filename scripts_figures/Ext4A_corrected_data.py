# %%
# IMPORTS
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths

from src import modelling_bio_beta as model
from src import batch_correction as bc
import os.path



N_CORES = 15

N_SITES = None
N_PARTICIPANTS = None

EXT_DSET_NAME = 'downsyndrome' # external dataset name
REF_DSET_NAME = 'wave3' # reference datset name
