import matplotlib.pyplot as plt
import numpy as np
from utils2 import *
from BGMCA3 import BGMCA as bgmca3
import copy as cp
from tqdm import tqdm

try:
################# Main test script
    totalSize = 2000
    minSizeBlock = 200
    divisors = list(divisorGenerator(totalSize))
    divisors = np.array(divisors)
    divisors = divisors[divisors>=minSizeBlock]
    # print('divisors: ' + str(divisors))

    data_path = '../data/'
    plots_path = '../plots/'

    divisors = np.array([200])
    alphas = np.array([ 0.001, 0.01, 50., 100.])
    alpha_gg = np.array([0.25, 0.5, 0.85, 1.5])

    numIts = 10
    n_obs = 10
    n_s = 5

    ca_final_sB = np.zeros([2,len(alphas),numIts,len(alpha_gg)]) # [i,:,:] i=0 mean i=1 med


    title_str = "LAST2_Test_pics_alphas" + "_totalSize_" + str(totalSize) + "_testAlphas_" + str(len(alphas)) + \
    "_alphaGG_" + str(len(alpha_gg)) + "_minSizeBlock_" + str(minSizeBlock) +"_numIts_" + str(numIts) + "_n_obs_" \
    + str(n_obs) + "_n_s_" + str(n_s)

    for itGG in range(len(alpha_gg)):

        for it0 in tqdm(range(numIts)):

            X,X0,A0,S0,N = Make_Experiment_GG(n_s=n_s,n_obs=n_obs,t_samp=totalSize,noise_level=60.0,\
                                              dynamic=0,CondNumber=1,alpha=alpha_gg[itGG])


            for it1 in range(len(alphas)):

                numBlock = totalSize/divisors[0]
                sizeBlock = divisors[0]
                # print('divisor: ' + str(divisors[it1]))
                X_ = cp.deepcopy(X)

                Results_sB = bgmca3(X_,n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
                    BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
                    threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=0,optA=1,alpha_exp=alphas[it1])

                A_sB = Results_sB['mixmat']
                S_sB = Results_sB['sources']
                crit_sB = EvalCriterion(A0,S0,A_sB,S_sB) 
                ca_final_sB[0,it1,it0,itGG] = crit_sB['ca_mean']
                ca_final_sB[1,it1,it0,itGG] = crit_sB['ca_med']
                
                # if crit_sB['ca_mean'] > 0.1:
                #     print('crit_sB[ca_mean] > 0.1')
                # if crit_sB['ca_med'] > 0.1:
                #     print('crit_sB[ca_med] > 0.1')           

    print('***********************************')            
    print('-10*np.log10(ca_final_sB): ')
    print(-10*np.log10(ca_final_sB))


    title_str_pdf = plots_path + title_str + ".pdf"

    # Save variables
    title_ca_final_sB = data_path + 'ca_final_sB' + title_str  
    np.save(title_ca_final_sB,ca_final_sB)


    ca_final_sB_mean = np.mean(ca_final_sB,axis=2)
    ca_final_sB_med = np.median(ca_final_sB,axis=2)

    print('***********************************')
    print('-10*np.log10(ca_final_sB_mean): ')
    print(-10*np.log10(ca_final_sB_mean))
    print('***********************************')
    print('-10*np.log10(ca_final_sB_med): ')
    print(-10*np.log10(ca_final_sB_med))

    log_alphas = np.log10(alphas)

    ################# Ploting script

    # import matplotlib
    # matplotlib.rcParams.update({'font.size': 18})

    # log_ca_final_sB_mean = -10*np.log10(ca_final_sB_mean)
    # log_ca_final_sB_med = -10*np.log10(ca_final_sB_med)


    # fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    # plt.rc('text', usetex=True)

    # for itGG in range(len(alpha_gg)):
    #     # plt.plot(divisors, log_ca_final_sB_mean[0,:], '-o', label=r'$CA_{mean}(mean)(nw_SC)$')
    #     # plt.plot(divisors, log_ca_final_sB_mean[1,:], '-o', label=r'$CA_{med}(mean)(nw_SC)$')
         
    #     plt.plot(log_alphas, log_ca_final_sB_med[0,:,itGG], '--o', label=r'$CA_{mean}, \alpha_{exp} = ' + str(alpha_gg[itGG]) + r'$')
    #     plt.plot(log_alphas, log_ca_final_sB_med[1,:,itGG], '-o', label=r'$CA_{med}, \alpha_{exp} = ' + str(alpha_gg[itGG]) + r'$')

    # plt.xlabel('log10(alphas)')
    # plt.ylabel(r'CA performance $\left(-10 \log_{10}(CA)\right)$')
    # plt.title(r"Test pics alphas .SubBlockSize=200/2000. $Reg \ell_0$. wSC. SNR=$60$dB ")
    # plt.grid()
    # plt.legend()
    # plt.show()

    # fig.savefig(title_str_pdf, bbox_inches='tight')

except:

    print('***********************************')            
    print('-10*np.log10(ca_final_sB): ')
    print(-10*np.log10(ca_final_sB))


    # title_str = "Test_pics_alphas" + "_totalSize_" + str(totalSize) + "_minSizeBlock_" + str(minSizeBlock) +\
    #     "_numIts_" + str(numIts) + "_n_obs_" + str(n_obs) + "_n_s_" + str(n_s)

    title_str_pdf = title_str + ".pdf"
    error_str = 'ERROR_'

    # Save variables
    title_ca_final_sB = data_path + error_str + 'ca_final_sB' + title_str  
    np.save(title_ca_final_sB,ca_final_sB)

