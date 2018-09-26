import matplotlib.pyplot as plt
import numpy as np
from utils2 import *
import copy as cp
from tqdm import tqdm
from BGMCA3 import BGMCA as bgmca3
from misc_bgmca2 import *


data_path = '../data/'
plots_path = '../plots/'

GG_mapping_param = np.array([ 4.5828 , -8.5807 ,  5.2610])

# Initialization
totalSize = 4000
divisors = np.array([400])
numBlock = totalSize/divisors[0]
sizeBlock = divisors[0]

numIts = 25

n_obs = 5
n_s = 5
rho_est = 0.9

alpha_est = GG_mapping_param[0]*(rho_est**2) + GG_mapping_param[1]*rho_est \
                + GG_mapping_param[2]
# alpha_est = 0.02

# rho_test = np.array([0.2,0.4,0.8,1.5])
alpha_test = np.array([25.])

rho_original = 0.4
alpha_original = GG_mapping_param[0]*(rho_original**2) + GG_mapping_param[1]*rho_original \
                + GG_mapping_param[2]

print('***********************************')   
print 'rho_est'
print rho_est
print 'alpha_est'
print alpha_est
print 'rho_original'
print rho_original
print 'alpha_original'
print alpha_original


ca_final_sB1 = np.zeros([2,len(alpha_test),numIts])
ca_final_sB0 = np.zeros([2,len(alpha_test),numIts])

print('***********************************')   
print('totalSize: ' + str(totalSize))
print('divisors: ' + str(divisors))
print('numIts: ' + str(numIts))
print('rho_original: ' + str(rho_original))
print('alpha_test: ' + str(alpha_test))
print('n_s: ' + str(n_s))
print('n_obs: ' + str(n_obs))
print('***********************************')   

title_str = "DEF25testValidation_alpha_est" + "_totalSize_" + str(totalSize) + "_numDivisors_" + str(len(divisors))+ "_n_obs_" + str(n_obs) \
    + "_n_s_" + str(n_s) + "_alpha_test_" + str(len(alpha_test)) + "_numIts_" + str(numIts)
print(title_str)
print('***********************************')   

for it_n in tqdm(range(numIts)):
    # Data generation    
    X,X0,A0,S0,N = Make_Experiment_GG(n_s=n_s,n_obs=n_obs,t_samp=totalSize,noise_level=80.0,\
                                      dynamic=0,CondNumber=1,alpha=rho_original)

    for it_1 in range(len(alpha_test)):
        alpha_est = alpha_test[it_1]   

        print 'alpha_est'
        print alpha_est 

        Results_sB1 = bgmca3(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
                BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
                threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=1,optA=1,alpha_exp=alpha_est)
        A_sB1 = Results_sB1['mixmat']
        S_sB1 = Results_sB1['sources']
        crit_sB1 = EvalCriterion_fast(A0,S0,A_sB1,S_sB1) 
        ca_final_sB1[0,it_1,it_n] = crit_sB1['ca_mean']
        ca_final_sB1[1,it_1,it_n] = crit_sB1['ca_med']
        
        Results_sB0 = bgmca3(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
                BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
                threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=0,optA=1,alpha_exp=alpha_est)
        A_sB0 = Results_sB0['mixmat']
        S_sB0 = Results_sB0['sources']
        crit_sB0 = EvalCriterion_fast(A0,S0,A_sB0,S_sB0) 
        ca_final_sB0[0,it_1,it_n] = crit_sB0['ca_mean']
        ca_final_sB0[1,it_1,it_n] = crit_sB0['ca_med']    


print('ca_final_sB1.shape:')
print(ca_final_sB1.shape)
print('ca_final_sB0.shape:')
print(ca_final_sB0.shape)

print('-10*np.log10(ca_final_sB1): ')
print(-10*np.log10(ca_final_sB1))
print('-10*np.log10(ca_final_sB0): ')
print(-10*np.log10(ca_final_sB0))

print('***************************')
print('***************************')

ca_final_sB0_mean = np.mean(ca_final_sB0,axis=2)
ca_final_sB1_mean = np.mean(ca_final_sB1,axis=2)

ca_final_sB0_med = np.median(ca_final_sB0,axis=2)
ca_final_sB1_med = np.median(ca_final_sB1,axis=2)

log_ca_final_sB0_mean = -10*np.log10(ca_final_sB0_mean)
log_ca_final_sB1_mean = -10*np.log10(ca_final_sB1_mean)

log_ca_final_sB0_med = -10*np.log10(ca_final_sB0_med)
log_ca_final_sB1_med = -10*np.log10(ca_final_sB1_med)

print 'log_ca_final_sB1_mean:'
print log_ca_final_sB1_mean
print 'log_ca_final_sB0_mean:'
print log_ca_final_sB0_mean
print 'log_ca_final_sB1_med:'
print log_ca_final_sB1_med
print 'log_ca_final_sB0_med:'
print log_ca_final_sB0_med


# Save variables
title_ca_final_sB0 = data_path + 'ca_final_sB0' + title_str  
np.save(title_ca_final_sB0,ca_final_sB0)
title_ca_final_sB1 = data_path + 'ca_final_sB1' + title_str  
np.save(title_ca_final_sB1,ca_final_sB1)


