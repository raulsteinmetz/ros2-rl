import math
import torch as T
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gpu():
    return T.device('cuda:0' if T.cuda.is_available() else 'cpu')


def fanin_init(size, fanin=None):
        fanin = fanin or size[0]
        v = 1./np.sqrt(fanin)
        return T.Tensor(size).uniform_(-v,v)
