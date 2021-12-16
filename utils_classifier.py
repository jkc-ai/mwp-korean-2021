import os
import sys
import datetime

import numpy as np
import pandas as pd

from enum import Enum

import torch
import json

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

class QType8(Enum):
    Arithmetic = 0
    Ordering = 1
    Combination = 2
    FindingNumber1 = 3
    FindingNumber2 = 4
    FindingNumber3 = 5
    Comparison = 6
    Geometry = 7
