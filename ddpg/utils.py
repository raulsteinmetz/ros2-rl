import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch as T

def get_gpu():
    return T.device('cuda:0' if T.cuda.is_available() else 'cpu')