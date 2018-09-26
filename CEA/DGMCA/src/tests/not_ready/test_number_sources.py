import sys
sys.path.append('../')
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib.pyplot as plt
import numpy as np
#import scipy.io as spio
#import scipy.signal as spsg
from src.misc_bgmca2 import *
from src.utils2 import *
from src.starlet import *
from src.BGMCA3 import BGMCA as bgmca3
from src.GMCA import GMCA as gmca_original
import copy as cp
#from tqdm import tqdm


#try:
################# Main test script
totalSize = 2000
minSizeBlock = 200
divisors = list(divisorGenerator(totalSize))
divisors = np.array(divisors)
divisors = divisors[divisors>=minSizeBlock]

data_path = '../test_data/'
plots_path = '../plots/'

n_sources = np.array([2,5,10,25])
n_obs = 50
numIts = 11

print('*******************************************')
print('divisors: ' + str(divisors))
print('n_sources: ' + str(n_sources))
print('numIts: ' + str(numIts))
print('n_obs: ' + str(n_obs))
print('*******************************************')

ca_final_sB = np.zeros([2,len(divisors),len(n_sources),numIts]) # [i,:,:,:] i=0 mean i=1 med
ca_final_tot = np.zeros([2,len(n_sources),numIts]) # [i,:,:] i=0 mean i=1 med


for it_n in range(len(n_sources)):
    n_s = n_sources[it_n]
#    n_obs = n_s

    for it0 in range(numIts):
        X,X0,A0,S0,N = Make_Experiment_GG(n_s=n_s,n_obs=n_obs,t_samp=totalSize,noise_level=60.0,dynamic=0,CondNumber=1,alpha=0.5)

        Results_original = gmca_original(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,\
            Aposit=False,BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,threshOpt=1,SCOpt=1)


        A_original = Results_original['mixmat']
        S_original = Results_original['sources']
        crit_original = EvalCriterion(A0,S0,A_original,S_original) 
        ca_final_tot[0,it_n,it0] = crit_original['ca_mean']
        ca_final_tot[1,it_n,it0] = crit_original['ca_med']

        if crit_original['ca_mean'] > 0.1:
            print('crit_original[ca_mean] > 0.1')
        if crit_original['ca_med'] > 0.1:
            print('crit_original[ca_med] > 0.1')    

        for it1 in range(len(divisors)):
            numBlock = totalSize/divisors[it1]
            sizeBlock = divisors[it1]

            Results_sB = bgmca3(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
                BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
                threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=0,optA=1,alpha_exp=2.)

            A_sB = Results_sB['mixmat']
            S_sB = Results_sB['sources']
            crit_sB = EvalCriterion(A0,S0,A_sB,S_sB) 
            ca_final_sB[0,it1,it_n,it0] = crit_sB['ca_mean']
            ca_final_sB[1,it1,it_n,it0] = crit_sB['ca_med']
            
            if crit_sB['ca_mean'] > 0.1:
                print('crit_sB[ca_mean] > 0.1')
            if crit_sB['ca_med'] > 0.1:
                print('crit_sB[ca_med] > 0.1')           


print('ca_final_tot: ')
print(ca_final_tot)
print('ca_final_sB: ')
print(ca_final_sB)


title_str = "Test_number_sources" + "_totalSize_" + str(totalSize) + "_minSizeBlock_" + str(minSizeBlock) +\
    "_numIts_" + str(numIts) + "_n_sources_" + str(len(n_sources))

title_str_pdf = plots_path + title_str + ".pdf"

# Save variables
title_ca_final_sB = data_path + 'ca_final_sB' + title_str  
np.save(title_ca_final_sB,ca_final_sB)
title_ca_final_tot = data_path + 'ca_final_tot' + title_str  
np.save(title_ca_final_tot,ca_final_tot)


ca_final_sB_mean = np.mean(ca_final_sB,axis=3)
ca_final_tot_mean = np.mean(ca_final_tot,axis=2)
ca_final_sB_med = np.median(ca_final_sB,axis=3)
ca_final_tot_med = np.median(ca_final_tot,axis=2)


################# Ploting script
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

log_ca_final_sB_mean = -10*np.log10(ca_final_sB_mean)
log_ca_final_sB_med = -10*np.log10(ca_final_sB_med)
log_ca_final_tot_mean = -10*np.log10(ca_final_tot_mean)
log_ca_final_tot_med = -10*np.log10(ca_final_tot_med)


    # fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    # plt.rc('text', usetex=True)

    # for it_n in range(len(n_sources)):
    #     # plt.plot(divisors, log_ca_final_tot_med[0,it_n]*np.ones(len(divisors)), '--', label=r'$CA_{mean}^{total}, N_{sources} ='\
    #     #     + str(n_sources[it_n]) + r'$')
    #     plt.plot(divisors, log_ca_final_tot_med[1,it_n]*np.ones(len(divisors)), '--', label=r'$CA_{med}^{total}, N_{sources} ='\
    #         + str(n_sources[it_n]) + r'$')
         
    #     # plt.plot(divisors, log_ca_final_sB_med[0,:,it_n], '-o', label=r'$CA_{mean}, N_{sources} = ' \
    #     #     + str(n_sources[it_n]) + r'$')
    #     plt.plot(divisors, log_ca_final_sB_med[1,:,it_n], '-o', label=r'$CA_{med}, N_{sources} = ' \
    #         + str(n_sources[it_n]) + r'$')

    # plt.xlabel('Batch size')
    # plt.ylabel(r'CA performance $\left(-10 \log_{10}(CA)\right)$')
    # plt.title(r"Test number of sources. $Reg \ell_0$. wSC. SNR=$60$dB ")
    # plt.grid()
    # plt.legend()
    # plt.show()

    # fig.savefig(title_str_pdf, bbox_inches='tight')


# except:

#     print('ca_final_tot: ')
#     print(ca_final_tot)
#     print('ca_final_sB: ')
#     print(ca_final_sB)


#     title_str = "Test_number_sources" + "_totalSize_" + str(totalSize) + "_minSizeBlock_" + str(minSizeBlock) +\
#         "_numIts_" + str(numIts) + "_n_sources_" + str(len(n_sources))

#     title_str_pdf = plots_path + title_str + ".pdf"
#     error_str = 'ERROR_'

#     # Save variables
#     title_ca_final_sB = data_path + error_str + 'ca_final_sB' + title_str  
#     np.save(title_ca_final_sB,ca_final_sB)
#     title_ca_final_tot = data_path + error_str + 'ca_final_tot' + title_str  
#     np.save(title_ca_final_tot,ca_final_tot)





