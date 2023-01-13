import numpy as np
import pandas as pd
# import anndata as ad 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocess import Pool
import time
# from multiprocess import Pool
import anndata as ad
sns.set_theme()

# import custom code
import sys
sys.path.append("..")   # fix to import modules from root
from src.amdata import amdata
from src.utils import plot
from src.utils import get
from src.utils import multiprocess


PATH = "/afs/inf.ed.ac.uk/user/s17/s1768506/disk/methyl-pattern/data/hannum/hannum_data.h5ad"
HANNUM_PATH = "/afs/inf.ed.ac.uk/user/s17/s1768506/disk/methyl-pattern/data/hannum/hannum_data.h5ad"
GENSCOT_PATH = "/afs/inf.ed.ac.uk/user/s17/s1768506/disk/methyl-pattern/data/gs/genscot.h5ad"

# for chunking in multiprocess
CPU_COUNT = 15
N_CHUNKS = 5 # amount of chunks each cpu should complete
CHUNKIFY = CPU_COUNT*N_CHUNKS

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)