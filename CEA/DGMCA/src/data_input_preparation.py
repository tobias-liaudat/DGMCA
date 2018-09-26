import sys
sys.path.append('../')
import numpy as np
import scipy.io as spio
import scipy.signal as spsg
from starlet import *
from misc_bgmca2 import *
from utils2 import *
# import imageio
import copy as cp

data_path = '../files/input_data/'

sim_path = data_path + 'SyntheticAstroData.mat'
imp_data = spio.loadmat(sim_path)
A_big = imp_data[',mixmat']
S_big = imp_data['sources']


