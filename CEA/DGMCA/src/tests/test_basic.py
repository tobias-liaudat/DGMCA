import sys
sys.path.append('../')
import numpy as np
from utils2 import *
from misc_bgmca import *
from BGMCA3 import BGMCA as dgmca
from DGMCA import DGMCA as dgmca_new
from GMCA import GMCA as gmca
import copy as cp
from tqdm import tqdm
import time

data_path = '../../files/data/'
plots_path = '../../files/plots/'

totalSize = 4000
minSizeBlock = 500
divisors = list(divisorGenerator(totalSize))
divisors = np.array(divisors)
divisors = divisors[divisors>=minSizeBlock]

divisors = np.array([200])# 200, 250 ,400, 500, 1000, 2500])

n_s = 5
n_obs = 20
rho = 0.5
alpha_init = 2.
lineBlockSize = 5

numIts = 1

ca_final_sB0 = np.zeros([2,len(divisors),numIts])
ca_final_sB_totSC = np.zeros([2,numIts])
time_results = np.zeros([3,numIts])



print('*******************************************')
print('totalSize: ' + str(totalSize))
print('divisors: ' + str(divisors))
print('lineBlockSize: ' + str(lineBlockSize))
print('numIts: ' + str(numIts))
print('rho: ' + str(rho))
print('alpha_init: ' + str(alpha_init))
print('n_s: ' + str(n_s))
print('n_obs: ' + str(n_obs))
print('*******************************************')

title_str = "testing_" + "_totalSize_" + str(totalSize) + "_numDivisors_" + str(len(divisors))+ "_n_obs_" + str(n_obs) \
    + "_n_s_" + str(n_s) + "_numIts_" + str(numIts)
print("Test saving name:")
print(title_str)
print('*******************************************')

try:

    for it_n in tqdm(range(numIts)):

        X,X0,A0,S0,N = Make_Experiment_GG(n_s=n_s,n_obs=n_obs,t_samp=totalSize,noise_level=60.0,\
                                          dynamic=0,CondNumber=1,alpha=rho)
        time1 = time.time()
        Results_sB_totSC = gmca(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,\
                    Aposit=False,BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,threshOpt=1\
                    ,SCOpt=1)
        time_results[0,it_n] = time.time() - time1
        A_sB_totSC = Results_sB_totSC['mixmat']
        S_sB_totSC = Results_sB_totSC['sources']
        crit_sB_totSC = EvalCriterion_fast(A0,S0,A_sB_totSC,S_sB_totSC) 
        ca_final_sB_totSC[0,it_n] = crit_sB_totSC['ca_mean']
        ca_final_sB_totSC[1,it_n] = crit_sB_totSC['ca_med']


        for it1 in range(len(divisors)):

            numBlock = totalSize/divisors[it1]
            sizeBlock = divisors[it1]


            time1 = time.time()
            Results_sB0 = dgmca_new(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
                                BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
                                threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=1,optA=1,alpha_exp=alpha_init)
            time_results[2,it_n] = time.time() - time1
            A_sB0 = Results_sB0['mixmat']
            S_sB0 = Results_sB0['sources']
            crit_sB0 = EvalCriterion_fast(A0,S0,A_sB0,S_sB0) 
            ca_final_sB0[0,it1,it_n] = crit_sB0['ca_mean']
            ca_final_sB0[1,it1,it_n] = crit_sB0['ca_med']


    print('******************* Results ************************')
    print('-10*np.log10(ca_final_sB_totSC): ')
    print(-10*np.log10(ca_final_sB_totSC))
    print('-10*np.log10(ca_final_sB0): ')
    print(-10*np.log10(ca_final_sB0))
    print('*******************************************')

    # Save variables
    title_ca_final_sB0 = data_path + 'ca_final_sB0' + title_str  
    np.save(title_ca_final_sB0,ca_final_sB0)
    title_ca_final_sB_totSC = data_path + 'ca_final_sB_totSC' + title_str  
    np.save(title_ca_final_sB_totSC,ca_final_sB_totSC)

except:

    print('-10*np.log10(ca_final_sB_totSC): ')
    print(-10*np.log10(ca_final_sB_totSC))
    print('-10*np.log10(ca_final_sB0): ')
    print(-10*np.log10(ca_final_sB0))

    error_str = 'ERROR_'

    # Save variables
    title_ca_final_sB0 = data_path + error_str + 'ca_final_sB0' + title_str  
    np.save(title_ca_final_sB0,ca_final_sB0)
    title_ca_final_sB_totSC = data_path + error_str + 'ca_final_sB_totSC' + title_str  
    np.save(title_ca_final_sB_totSC,ca_final_sB_totSC)

