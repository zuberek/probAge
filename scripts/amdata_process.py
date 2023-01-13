%load_ext autoreload
%autoreload 2
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

INPUT_PATH = '../exports/wave3.h5ad'
META_DIR_PATH = '/afs/inf.ed.ac.uk/user/s17/s1768506/disk/methylation/genscot_meta/wave3'

amdata = ad.read_h5ad(INPUT_PATH, backed='r')

preprocess_func.merge_meta()