import sys
sys.path.append('../')
import numpy as np
from utils2 import divisorGenerator
from utils2 import Make_Experiment_GG
from misc_bgmca2 import *
from DGMCA import DGMCA as dgmca
from GMCA import GMCA as gmca
from data_input_preparation import data_input_preparation
import copy as cp
from tqdm import tqdm
import time

input_data_path = '../../files/input_data/'
data_path = '../../files/data/'
plots_path = '../../files/plots/'


Jdec = 2
dec_factor = 64
batch_size = 1024
n_s = 5 # Number of sources
n_obs = 250 # Number of observations
SNR_level = 60.0
normOpt = 1
alpha_init = 2. # Alpha parameter for the thresholding strategy
numIts = 1

data_DGMCA = data_input_preparation(Jdec=Jdec, dec_factor=dec_factor, batch_size=batch_size, n_obs=n_obs, SNR_level=SNR_level, normOpt=normOpt,\
	input_data_path=input_data_path, output_data_path=data_path)

# Extract the variables
X = data_DGMCA['X']
A0 = data_DGMCA['A0']
S0 = data_DGMCA['S0']
X_GMCA = data_DGMCA['X_GMCA']
S0_GMCA = data_DGMCA['S0_GMCA']
sizeBlock = data_DGMCA['batch_size']
n_s = data_DGMCA['n_s']
WNFactors = data_DGMCA['WNFactors']
X0_GMCA = data_DGMCA['X0']
normOpt = data_DGMCA['normOpt'] 
totalSize = X.shape[1] 

CA_DGMCA = np.zeros([2,numIts])
CA_GMCA = np.zeros([2,numIts])
time_GMCA = np.zeros([numIts])
time_DGMCA = np.zeros([numIts])


print('*******************************************')
print('totalSize: ' + str(totalSize))
print('numIts: ' + str(numIts))
print('alpha_init: ' + str(alpha_init))
print('n_s: ' + str(n_s))
print('n_obs: ' + str(n_obs))
print('*******************************************')

title_str = "test_basic_synthetic_data" + "_totalSize_" + str(totalSize) + "_batch_size_" + str(batch_size)+ "_n_obs_" + str(n_obs) \
    + "_n_s_" + str(n_s) + '_Jdec_' + str(Jdec) + "_numIts_" + str(numIts)
print("Test saving name:")
print(title_str)
print('*******************************************')


for it_n in tqdm(range(numIts)):

    # X,X0,A0,S0,N = Make_Experiment_GG(n_s=n_s,n_obs=n_obs,t_samp=totalSize,noise_level=60.0,\
    #                                   dynamic=0,CondNumber=1,alpha=rho)
    time1 = time.time()
    Results_sB_totSC = gmca(X_GMCA,n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,\
                Aposit=False,BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,threshOpt=1\
                ,SCOpt=1)
    time_GMCA[it_n] = time.time() - time1
    A_sB_totSC = Results_sB_totSC['mixmat']
    S_sB_totSC = Results_sB_totSC['sources']
    crit_sB_totSC = EvalCriterion_eval(A0,S0_GMCA,A_sB_totSC,S_sB_totSC) 
    CA_GMCA[0,it_n] = crit_sB_totSC['ca_mean']
    CA_GMCA[1,it_n] = crit_sB_totSC['ca_med']


    time1 = time.time()
    Results_sB0 = dgmca(X,n=n_s,mints=3,nmax=100,L0=1,verb=0,Init=0,BlockSize= None,Kmax=1.,AInit=None,tol=1e-6,\
    		subBlockSize=sizeBlock, SCOpt=1,alphaEstOpt=1,alpha_exp=alpha_init, J=Jdec,WNFactors=WNFactors,normOpt=normOpt)
    time_DGMCA[it_n] = time.time() - time1
    A_sB0 = Results_sB0['mixmat']
    S_sB0 = Results_sB0['sources']
    crit_sB0 = EvalCriterion_eval(A0,S0_GMCA,A_sB0,S_sB0) 
    CA_DGMCA[0,it_n] = crit_sB0['ca_mean']
    CA_DGMCA[1,it_n] = crit_sB0['ca_med']


# Some calculations to print afterwards
time_GMCA_mean = np.mean(time_GMCA,axis=0)
time_GMCA_total = np.sum(time_GMCA, axis=0)

time_DGMCA_mean = np.mean(time_DGMCA,axis=0)
time_DGMCA_total = np.sum(time_DGMCA, axis=0)

CA_GMCA_mean = np.mean(CA_GMCA,axis=1)
CA_DGMCA_mean = np.mean(CA_DGMCA,axis=1)
CA_GMCA_med = np.median(CA_GMCA,axis=1)
CA_DGMCA_med = np.median(CA_DGMCA,axis=1)

dB_CA_GMCA_mean = -10*np.log10(CA_GMCA_mean)
dB_CA_DGMCA_mean = -10*np.log10(CA_DGMCA_mean)
dB_CA_GMCA_med = -10*np.log10(CA_GMCA_med)
dB_CA_DGMCA_med = -10*np.log10(CA_DGMCA_med)

# Printing the results nicely
print(' ')
print('******************* Results ************************')
print('GMCA [CA(dB)]            / DGMCA [CA(dB)]')
print('GMCA mean time:  %f  / DGMCA mean time:  %f'%(time_GMCA_mean,time_DGMCA_mean))
print('GMCA total time: %f  / DGMCA total time: %f'%(time_GMCA_total,time_DGMCA_total))
print(' ')
print('**************** MEAN of experiments ***************************')
print('Total size = %d, CA_GMCA(med) = %f,  CA_GMCA(mean) = %f ;'%(totalSize,dB_CA_GMCA_mean[1],dB_CA_GMCA_mean[0]))
print('*******************************************')
print('Batch size = %d, CA_DGMCA(med) = %f,  CA_DGMCA(mean) = %f ;'%(batch_size,dB_CA_DGMCA_mean[1],dB_CA_DGMCA_mean[0]))
print('*******************************************')
print(' ')
print('**************** MEDIAN of experiments ***************************')
print('Total size = %d, CA_GMCA(med) = %f,  CA_GMCA(mean) = %f ;'%(totalSize,dB_CA_GMCA_med[1],dB_CA_GMCA_med[0]))
print('*******************************************')
print('Batch size = %d, CA_DGMCA(med) = %f,  CA_DGMCA(mean) = %f ;'%(batch_size,dB_CA_DGMCA_med[1],dB_CA_DGMCA_med[0]))
print('*******************************************')
print(' ')

# Save variables
title_CA_DGMCA = data_path + 'CA_DGMCA' + title_str  
np.save(title_CA_DGMCA,CA_DGMCA)
title_CA_GMCA = data_path + 'CA_GMCA' + title_str  
np.save(title_CA_GMCA,CA_GMCA)









