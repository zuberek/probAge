import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
import anndata as ad
# sns.set_theme(style='ticks')

# import custom code
import sys
sys.path.append("..")   # fix to import modules from root
import src.amdata.amdata as amdata_src
from src.utils import plot

sns_colors = sns.color_palette().as_hex()
