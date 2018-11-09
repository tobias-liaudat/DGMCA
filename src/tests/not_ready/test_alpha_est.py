import sys
import matplotlib.pyplot as plt
import numpy as np
from utils2 import *
# from bgmca7 import GMCA as gmca_block
# from bgmca7_bis import GMCA as gmca_block_bis
from BGMCA3 import BGMCA as bgmca3
import copy as cp
from tqdm import tqdm
import scipy as sp
from misc_bgmca2 import *
import time

n_s = 5
n_obs = 5
totalSize = 10000
rho = 1.
sizeBlock = 1000
alpha_init = 0.05

data_path = '../data/'
plots_path = '../plots/'

numIts = 11

ca_final_sB1 = np.zeros([2,numIts])
ca_final_sB1_bis = np.zeros([2,numIts])
ca_final_sB1_bis2 = np.zeros([2,numIts])
ca_final_sB0 = np.zeros([2,numIts])
time_results = np.zeros([4,numIts])


print('totalSize: ' + str(totalSize))
print('sizeBlock: ' + str(sizeBlock))
print('numIts: ' + str(numIts))
print('rho: ' + str(rho))
print('alpha_init: ' + str(alpha_init))

for it_n in tqdm(range(numIts)):

	X,X0,A0,S0,N = Make_Experiment_GG(n_s=n_s,n_obs=n_obs,t_samp=totalSize,noise_level=100.0,\
	                                  dynamic=0,CondNumber=1,alpha=rho)

	# time1 = time.time()
	# Results_sB1_bis = gmca_block_bis(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=120,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
	#                     BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
	#                     threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=1,optA=0,alpha_exp=alpha_init)
	# time_results[0,it_n] = time.time() - time1
	# A_sB1_bis = Results_sB1_bis['mixmat']
	# S_sB1_bis = Results_sB1_bis['sources']
	# crit_sB1_bis = EvalCriterion_fast(A0,S0,A_sB1_bis,S_sB1_bis) 
	# ca_final_sB1_bis[0,it_n] = crit_sB1_bis['ca_mean']
	# ca_final_sB1_bis[1,it_n] = crit_sB1_bis['ca_med']

	time1 = time.time()
	Results_sB1_bis2 = bgmca3(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=120,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
	                    BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
	                    threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=1,optA=1,alpha_exp=alpha_init)
	time_results[1,it_n] = time.time() - time1
	A_sB1_bis2 = Results_sB1_bis2['mixmat']
	S_sB1_bis2 = Results_sB1_bis2['sources']
	crit_sB1_bis2 = EvalCriterion_fast(A0,S0,A_sB1_bis2,S_sB1_bis2) 
	ca_final_sB1_bis2[0,it_n] = crit_sB1_bis2['ca_mean']
	ca_final_sB1_bis2[1,it_n] = crit_sB1_bis2['ca_med']
	

	# time1 = time.time()
	# Results_sB1 = gmca_block(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=120,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
	#                     BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
	#                     threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=1,alpha_exp=alpha_init)
	# time_results[2,it_n] = time.time() - time1
	# A_sB1 = Results_sB1['mixmat']
	# S_sB1 = Results_sB1['sources']
	# crit_sB1 = EvalCriterion_fast(A0,S0,A_sB1,S_sB1) 
	# ca_final_sB1[0,it_n] = crit_sB1['ca_mean']
	# ca_final_sB1[1,it_n] = crit_sB1['ca_med']

	time1 = time.time()
	Results_sB0 = bgmca3(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=120,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
	                    BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
	                    threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=0,alpha_exp=alpha_init)
	time_results[3,it_n] = time.time() - time1
	A_sB0 = Results_sB0['mixmat']
	S_sB0 = Results_sB0['sources']
	crit_sB0 = EvalCriterion_fast(A0,S0,A_sB0,S_sB0) 
	ca_final_sB0[0,it_n] = crit_sB0['ca_mean']
	ca_final_sB0[1,it_n] = crit_sB0['ca_med']


# log_ca_final_sB1_bis = -10*np.log10(ca_final_sB1_bis)
log_ca_final_sB1_bis2 = -10*np.log10(ca_final_sB1_bis2)
# log_ca_final_sB1 = -10*np.log10(ca_final_sB1)
log_ca_final_sB0 = -10*np.log10(ca_final_sB0)

mean_time_results = np.mean(time_results,axis=1)


# print('Test alphaEstOpt=1 online. Mean execution time = %f'%(mean_time_results[2]))
# print 'log_ca_final_sB1:'
# print np.median(log_ca_final_sB1,axis=1)

# print('Test alphaEstOpt=1 warm start. New A. Mean execution time = %f'%(mean_time_results[0]))
# print 'log_ca_final_sB1_bis:'
# print np.median(log_ca_final_sB1_bis,axis=1)

print('Test alphaEstOpt=1 warm start. Old A. Mean execution time = %f'%(mean_time_results[1]))
print '(Mean) log_ca_final_sB1_bis2:'
print np.mean(log_ca_final_sB1_bis2,axis=1)
print '(Median) log_ca_final_sB1_bis2:'
print np.median(log_ca_final_sB1_bis2,axis=1)

print('Test alphaEstOpt=0. Mean execution time = %f'%(mean_time_results[3]))
print '(Mean) log_ca_final_sB0:'
print np.mean(log_ca_final_sB0,axis=1)
print '(Median) log_ca_final_sB0:'
print np.median(log_ca_final_sB0,axis=1)

title_str = "Test_alpha_est" + "_totalSize_" + str(totalSize) + "_SizeBlock_" + str(sizeBlock) +\
    "_numIts_" + str(numIts) + '_alpha_init_' + str(alpha_init) + '_rho_' + str(rho)

# Save variables
# title_ca_final_sB1 = data_path + 'ca_final_sB1' + title_str  
# np.save(title_ca_final_sB1,ca_final_sB1)

# title_ca_final_sB1_bis = data_path + 'ca_final_sB1_bis' + title_str  
# np.save(title_ca_final_sB1_bis,ca_final_sB1_bis)

title_ca_final_sB1_bis2 = data_path + 'ca_final_sB1_bis2' + title_str  
np.save(title_ca_final_sB1_bis2,ca_final_sB1_bis2)

title_ca_final_sB0 = data_path + 'ca_final_sB0' + title_str  
np.save(title_ca_final_sB0,ca_final_sB0)

# title_time_results = data_path + 'time_results' + title_str  
# np.save(title_time_results,time_results)

