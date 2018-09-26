import matplotlib.pyplot as plt
import numpy as np
from utils2 import *
from misc_bgmca import *
from GMCA import GMCA as gmca_original_SC
from BLGMCA4 import BLGMCA as blgmca4
from BGMCA3 import BGMCA as bgmca3
import copy as cp
from tqdm import tqdm
import time

data_path = '../data/'
plots_path = '../plots/'

try:
    totalSize = 4000
    minSizeBlock = 500
    divisors = list(divisorGenerator(totalSize))
    divisors = np.array(divisors)
    divisors = divisors[divisors>=minSizeBlock]

    divisors = np.array([200, 250 ,400, 500, 1000, 2000, 4000])
    lineBlockSize = np.array([10,30])#5,8,10,20])#,12,14,16,18,20,25])#5, 6, 7, 8, 9, 10,20])

    n_s = 5
    n_obs = 30

    rho = 0.5
    alpha_init = 2.

    numIts = 5

    ca_final_sB1 = np.zeros([2,len(divisors),len(lineBlockSize),numIts])
    # ca_final_sB2 = np.zeros([2,len(lineBlockSize),numIts])
    ca_final_sB0 = np.zeros([2,len(divisors),numIts])
    ca_final_sB_totSC = np.zeros([2,numIts])
    time_results = np.zeros([3,numIts])

    print('totalSize: ' + str(totalSize))
    print('divisors: ' + str(divisors))
    print('lineBlockSize: ' + str(lineBlockSize))
    print('numIts: ' + str(numIts))
    print('rho: ' + str(rho))
    print('alpha_init: ' + str(alpha_init))
    print('n_s: ' + str(n_s))
    print('n_obs: ' + str(n_obs))


    title_str = "DEF6_Test_col_lines" + "_totalSize_" + str(totalSize) + "_numLines_" + str(len(lineBlockSize)) + "_numDivisors_" + str(len(divisors))\
     + "_n_obs_" + str(n_obs) + "_n_s_" + str(n_s)  + "_numIts_" + str(numIts) 

    for it_n in tqdm(range(numIts)):

        X,X0,A0,S0,N = Make_Experiment_GG(n_s=n_s,n_obs=n_obs,t_samp=totalSize,noise_level=60.0,\
                                          dynamic=0,CondNumber=1,alpha=rho)
        time1 = time.time()
        Results_sB_totSC = gmca_original_SC(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,\
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
            # print 'numBlock'
            # print numBlock
            # print 'sizeBlock'
            # print sizeBlock

            time1 = time.time()
            Results_sB0 = bgmca3(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
                                BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
                                threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=1,optA=1,alpha_exp=alpha_init)
            time_results[2,it_n] = time.time() - time1
            A_sB0 = Results_sB0['mixmat']
            S_sB0 = Results_sB0['sources']
            crit_sB0 = EvalCriterion_fast(A0,S0,A_sB0,S_sB0)
            # print 'crit_sB0'
            # print crit_sB0
            ca_final_sB0[0,it1,it_n] = crit_sB0['ca_mean']
            ca_final_sB0[1,it1,it_n] = crit_sB0['ca_med']


            for it2 in range(len(lineBlockSize)):

                time1 = time.time()
                Results_sB1 = blgmca4(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
                                    BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
                                    lineBlockSize=lineBlockSize[it2],threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=1,optA=1,alpha_exp=alpha_init)
                time_results[1,it_n] = time.time() - time1
                A_sB1 = Results_sB1['mixmat']
                S_sB1 = Results_sB1['sources']
                crit_sB1 = EvalCriterion_fast(A0,S0,A_sB1,S_sB1)
                # print 'crit_sB1'
                # print crit_sB1
                ca_final_sB1[0,it1,it2,it_n] = crit_sB1['ca_mean']
                ca_final_sB1[1,it1,it2,it_n] = crit_sB1['ca_med']



                # time1 = time.time()
                # Results_sB2 = blgmca3(cp.deepcopy(X),n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
                #                     BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
                #                     lineBlockSize=lineBlockSize[it1],threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=1,optA=1,alpha_exp=alpha_init)
                # time_results[1,it_n] = time.time() - time1
                # A_sB2 = Results_sB2['mixmat']
                # S_sB2 = Results_sB2['sources']
                # crit_sB2 = EvalCriterion_fast(A0,S0,A_sB2,S_sB2) 
                # ca_final_sB2[0,it1,it_n] = crit_sB2['ca_mean']
                # ca_final_sB2[1,it1,it_n] = crit_sB2['ca_med']




    print('-10*np.log10(ca_final_sB_totSC): ')
    print(-10*np.log10(ca_final_sB_totSC))
    print('-10*np.log10(ca_final_sB0): ')
    print(-10*np.log10(ca_final_sB0))
    print('-10*np.log10(ca_final_sB1): ')
    print(-10*np.log10(ca_final_sB1))
    # print('-10*np.log10(ca_final_sB2): ')
    # print(-10*np.log10(ca_final_sB2))


    #     # print('-10*np.log10(ca_final_sB_totSC): ')
    #     # print(-10*np.log10(ca_final_sB_totSC))

    # Save variables
    title_ca_final_sB0 = data_path + 'ca_final_sB0' + title_str  
    np.save(title_ca_final_sB0,ca_final_sB0)
    title_ca_final_sB1 = data_path + 'ca_final_sB1' + title_str  
    np.save(title_ca_final_sB1,ca_final_sB1)
    # title_ca_final_sB2 = data_path + 'ca_final_sB2' + title_str  
    # np.save(title_ca_final_sB2,ca_final_sB2)
    title_ca_final_sB_totSC = data_path + 'ca_final_sB_totSC' + title_str  
    np.save(title_ca_final_sB_totSC,ca_final_sB_totSC)

#     title_time_results = data_path + 'time_results' + title_str  
#     np.save(title_time_results,time_results)


#     ca_final_sB0_mean = np.mean(ca_final_sB0,axis=1)
#     ca_final_sB1_mean = np.mean(ca_final_sB1,axis=2)
#     # ca_final_sB2_mean = np.mean(ca_final_sB2,axis=2)
#     ca_final_sB_totSC_mean = np.mean(ca_final_sB_totSC,axis=1)

#     ca_final_sB0_med = np.median(ca_final_sB0,axis=1)
#     ca_final_sB1_med = np.median(ca_final_sB1,axis=2)
#     # ca_final_sB2_med = np.median(ca_final_sB2,axis=2)
#     ca_final_sB_totSC_med = np.median(ca_final_sB_totSC,axis=1)

#     log_ca_final_sB0_mean = -10*np.log10(ca_final_sB0_mean)
#     log_ca_final_sB1_mean = -10*np.log10(ca_final_sB1_mean)
#     # log_ca_final_sB2_mean = -10*np.log10(ca_final_sB2_mean)
#     log_ca_final_sB_totSC_mean = -10*np.log10(ca_final_sB_totSC_mean)

#     log_ca_final_sB0_med = -10*np.log10(ca_final_sB0_med)
#     log_ca_final_sB1_med = -10*np.log10(ca_final_sB1_med)
#     # log_ca_final_sB2_med = -10*np.log10(ca_final_sB2_med)
#     log_ca_final_sB_totSC_med = -10*np.log10(ca_final_sB_totSC_med)


#     ################# Ploting script
#     import matplotlib
#     matplotlib.rcParams.update({'font.size': 18})
#     fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
#     plt.rc('text', usetex=True)

#     plt.plot(lineBlockSize, log_ca_final_sB_totSC_med[0]*np.ones(len(lineBlockSize)), '-.+', label=r'$CA_{mean}(gmca)$')
#     plt.plot(lineBlockSize, log_ca_final_sB_totSC_med[1]*np.ones(len(lineBlockSize)), '-.+', label=r'$CA_{med}(gmca)$')

#     plt.plot(lineBlockSize, log_ca_final_sB0_med[0]*np.ones(len(lineBlockSize)), '--*', label=r'$CA_{mean}(bgmca)$')
#     plt.plot(lineBlockSize, log_ca_final_sB0_med[1]*np.ones(len(lineBlockSize)), '-*', label=r'$CA_{med}(bgmca)$')

#     plt.plot(lineBlockSize, log_ca_final_sB1_med[0,:], '--o', label=r'$CA_{mean}(bgmca, lineBS)$')
#     plt.plot(lineBlockSize, log_ca_final_sB1_med[1,:], '-o', label=r'$CA_{med}(bgmca, lineBS)$')

#     # plt.plot(lineBlockSize, log_ca_final_sB2_med[0,:], '--o', label=r'$CA_{mean}(bgmca bis, lineBS)$')
#     # plt.plot(lineBlockSize, log_ca_final_sB2_med[1,:], '-o', label=r'$CA_{med}(bgmca bis, lineBS)$')

#     plt.xlabel('Batch size')
#     plt.ylabel(r'CA performance $\left(-10 \log_{10}(CA)\right)$')
#     plt.title(r'Test line Numbers. $Reg \ell_0$. nwSC. SNR=$60$dB. $\rho = 0.5$. $\alpha_{exp}=2$. $numIts=' \
#             + str(numIts) + r'$  $n_s = ' + str(n_s) + r'$ $ n_{obs} = ' + str(n_obs) + r'$. $blockSize = ' \
#             + str(sizeBlock) + r'$.' )
#     plt.grid()
#     plt.legend()
#     plt.show()

#     title_str_pdf = plots_path + title_str + ".pdf"

#     fig.savefig(title_str_pdf, bbox_inches='tight')


except:

    print('-10*np.log10(ca_final_sB_totSC): ')
    print(-10*np.log10(ca_final_sB_totSC))
    print('-10*np.log10(ca_final_sB1): ')
    print(-10*np.log10(ca_final_sB1))
#     # print('-10*np.log10(ca_final_sB2): ')
#     # print(-10*np.log10(ca_final_sB2))
    print('-10*np.log10(ca_final_sB0): ')
    print(-10*np.log10(ca_final_sB0))


#     title_str = "Test_lineNumber" + "_totalSize_" + str(totalSize) + "_numLines_" + str(len(lineBlockSize))+ "_n_obs_" + str(n_obs) \
#         + "_n_s_" + str(n_s) + '_sizeBlock_' + str(sizeBlock) + "_numIts_" + str(numIts) 

#     title_str_pdf = plots_path + title_str + ".pdf"

    error_str = 'ERROR_'
    # Save variables
    title_ca_final_sB0 = data_path + error_str + 'ca_final_sB0' + title_str  
    np.save(title_ca_final_sB0,ca_final_sB0)
    title_ca_final_sB1 = data_path + error_str + 'ca_final_sB1' + title_str  
    np.save(title_ca_final_sB1,ca_final_sB1)
#     # title_ca_final_sB2 = data_path +  error_str + 'ca_final_sB2' + title_str  
#     # np.save(title_ca_final_sB2,ca_final_sB2)
    title_ca_final_sB_totSC = data_path + error_str + 'ca_final_sB_totSC' + title_str  
    np.save(title_ca_final_sB_totSC,ca_final_sB_totSC)
#     title_time_results = data_path + error_str + 'time_results' + title_str  
#     np.save(title_time_results,time_results)