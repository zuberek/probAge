import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocess import Pool
from multiprocess import cpu_count
import anndata as ad
from functools import partial
import arviz as az

import json

# progress bar
from IPython.display import clear_output, DisplayHandle
def update_patch(self, obj):
    clear_output(wait=True)
    self.display(obj)
DisplayHandle.update = update_patch

# sns.set_theme(style='ticks')

# import custom code
import sys
sys.path.append("..")   # fix to import modules from root
import src.amdata.amdata as amdata_src
from src.utils import plot
from src import paths

plt.rcParams['svg.fonttype'] = 'none'

sns_colors = sns.color_palette().as_hex()
# https://nanx.me/ggsci/reference/pal_npg.html
colors = [
    '#4DBBD5FF',    # blue
    '#E64B35FF',    # red
    '#00A087FF',    # dark green
    '#F39B7FFF',    # orenginh
    '#3C5488FF',    # dark blue
    '#8491B4FF',
    '#91D1C2FF',
    '#DC0000FF',
    '#B09C85FF',    # light grey
    '#7E6148FF',
]
sns.set_palette(sns.color_palette(colors))

CON_PALLETE = sns.color_palette("blend:#E64B35,#4DBBD5")
# plt.rc("axes.spines", top=False, right=False)


import logging
logger = logging.getLogger('pymc')
logger.propagate = False
logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")
