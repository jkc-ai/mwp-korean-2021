import os
import sys
import json
import datetime

import numpy as np
import pandas as pd

from enum import Enum


class QType8(Enum):
    Arithmetic = 0
    Ordering = 1
    Combination = 2
    FindingNumber1 = 3
    FindingNumber2 = 4
    FindingNumber3 = 5
    Comparison = 6
    Geometry = 7
