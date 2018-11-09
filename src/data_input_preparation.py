# Import modules
import sys
sys.path.append('../')
import numpy as np
import scipy.io as spio
from starlet import *
from misc_bgmca2 import *
from utils2 import *
import copy as cp

def data_input_preparation(Jdec=2, dec_factor=16, batch_size=2048, n_obs=250, SNR_level=60.0, normOpt=1,\
 	input_data_path='../files/input_data/', output_data_path = '../files/data/'):
    """
    Jdec : Wavelet decomposition levels
    dec_factor : Subsampling factor for each dimension of the image
    batch_size : Batch size to use in the DGMCA algorithm
    n_obs : Number of observations to use
    SNR_level : SNR level desired for the observations
    normOpt      : Normalize wavelet decomposition levels (for J>0)
                    # 0 --> Deactivated
                    # 1 --> Activated
    """
    # input_data_path = '../files/input_data/'
    # output_data_path = '../files/data/'

    sim_path = input_data_path + 'SyntheticAstroData.mat'
    imp_data = spio.loadmat(sim_path)
    A_big = imp_data[',mixmat']
    S_big = imp_data['sources']

    data_DGMCA = input_prep(A_big,S_big,dec_factor,Jdec,normOpt,batch_size,n_obs,SNR_level)
    str_data_DGMCA = output_data_path + 'dataGMCA_decf' + str(dec_factor) + '_J_' + str(Jdec) + '_norm_' + str(normOpt) +\
        '_bsize_' + str(batch_size) + '_nobs_' + str(n_obs)
    spio.savemat(str_data_DGMCA,data_DGMCA)

    print('*******************************************')
    print("Data saving string:")
    print(str_data_DGMCA)
    print('*******************************************')

    return data_DGMCA


