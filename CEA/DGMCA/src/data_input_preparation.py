import sys
sys.path.append('../')
import numpy as np
import scipy.io as spio
from starlet import *
from misc_bgmca2 import *
from utils2 import *
import copy as cp

input_data_path = '../files/input_data/'
output_data_path = '../files/data/'

sim_path = input_data_path + 'SyntheticAstroData.mat'
imp_data = spio.loadmat(sim_path)
A_big = imp_data[',mixmat']
S_big = imp_data['sources']


Jdec = 2 # Wavelet decomposition levels
dec_factor = 32 # Subsampling for each dimension of the image
normOpt = 1
batch_size = 2048 # 1024 # 8192 # 4096 # 2048
n_obs = 250
SNR_level = 60.0
data_GMCA = input_prep(A_big,S_big,dec_factor,Jdec,normOpt,batch_size,n_obs,SNR_level)
str_data_GMCA = output_data_path + 'dataGMCA_decf' + str(dec_factor) + '_J_' + str(Jdec) + '_norm_' + str(normOpt) +\
    '_bsize_' + str(batch_size) + '_nobs_' + str(n_obs)
spio.savemat(str_data_GMCA,data_GMCA)

print('*******************************************')
print("Data saving string:")
print(str_data_GMCA)
print('*******************************************')