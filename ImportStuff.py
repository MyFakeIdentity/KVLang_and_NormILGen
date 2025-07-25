import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os

# matplotlib.use('TkAgg')

ROOT_DIR = "Your\\root\\directory"
# ROOT_DIR = "/content/drive/MyDrive/Masters Project"
LANGUAGES_FILE = "Languages/Datasets/languages.json"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
plt.ion()
# DEVICE = torch.device("cpu")
