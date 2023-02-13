import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
import anndata as ad
import json

from functools import partial
# sns.set_theme(style='ticks')

# import custom code
import sys
sys.path.append("..")   # fix to import modules from root
import src.amdata.amdata as amdata_src
from src.utils import plot
plt.rcParams['svg.fonttype'] = 'none'

sns_colors = sns.color_palette().as_hex()
colors = [
    '#4DBBD5FF',
    '#E64B35FF',
    '#00A087FF',
    '#3C5488FF',
    '#F39B7FFF',
    '#8491B4FF',
    '#91D1C2FF',
    '#DC0000FF',
    '#7E6148FF',
    '#B09C85FF',
]
sns.set_palette(sns.color_palette(colors))


