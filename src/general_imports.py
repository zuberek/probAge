import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool
import anndata as ad
# sns.set_theme(style='ticks')

# import custom code
import sys
sys.path.append("..")   # fix to import modules from root
from src.amdata import amdata
from src.utils import plot
from src import preprocess_func
