import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from utils2 import *
from misc_bgmca2 import *
# from decBgmca_v3_rl1 import decBGMCA as decBGMCA
# from decBgmca_v3 import decBGMCA as decBGMCA2
# from decBLgmca_v2 import decBLGMCA as decBLGMCA
# from gmca3 import GMCA as GMCA
# from bgmca_v2 import GMCA as BGMCA
from BLGMCA2_bis import BLGMCA as blgmca2_bis
from GMCA import GMCA as gmca_original_SC
from BLGMCA4 import BLGMCA as blgmca4
from BGMCA3 import BGMCA as bgmca3
import copy as cp
from tqdm import tqdm
import time

data_path = '../input_data/'
save_data_path = '../data/'
plots_path = '../plots/'

# totalSize = 4500
# minSizeBlock = 500
# divisors = list(divisorGenerator(totalSize))
# divisors = np.array(divisors)
# divisors = divisors[divisors>=minSizeBlock]

divisors = np.array([256,512,1024,2048,4096,8192])
line_divs = np.array([25])#25,50])

bgmca_bool = 1

n_s = 5
n_obs = 250
rho = 0.5
alpha_init = 2.
# lineBlockSize = 25
# sizeBlock = 8192 # 8192 # 4096 # 2048 # 1024
J = 2
decF = 16 # 4 # 8 # 16 # 32
normOpt_test = 1
opt_GLOB_id_obs = 1

numIts = 1

ca_final_GMCA = np.zeros([2,len(divisors),numIts])
ca_final_DGMCA = np.zeros([2,len(divisors),numIts])
ca_final_xDGMCA = np.zeros([2,len(divisors),len(line_divs),numIts])
# ca_final_sB3 = np.zeros([2,len(divisors),numIts])
# ca_final_sB_totSC = np.zeros([2,numIts])
# time_results = np.zeros([3,numIts])

optGMCA         = 1

optDecBGMCA     = 1

optDecBLGMCA    = 1


for it_n in tqdm(range(numIts)):

    for it_div in range(len(divisors)):
        sizeBlock = divisors[it_div]

        if J > 0:
            # data_str = 'dataGMCA_decf' + str(decF) + '_J_' + str(J) + '_norm_' + str(normOpt_test)
            # data_str = 'dataGMCA_decf' + str(decF) + '_J_' + str(J) + '_norm_' + str(normOpt_test) + \
            #     '_bsize_' + str(sizeBlock)
            # data_str = 'dataGMCA_decf' + str(decF) + '_J_' + str(J) + '_norm_' + str(normOpt_test) + \
            #     '_bsize_' + str(sizeBlock) + '_nobs_' + str(n_obs)
            data_str = 'dataGMCA_decf' + str(decF) + '_J_' + str(J) + '_norm_' + str(normOpt_test) + \
            '_bsize_' + str(sizeBlock) + '_nobs_' + str(n_obs) + '_optGLOBidObs_' + str(opt_GLOB_id_obs)

            input_data = spio.loadmat(data_path + data_str)
            X = input_data['X']
            A0 = input_data['A0']
            S0 = input_data['S0']
            X_GMCA = input_data['X_GMCA']
            S0_GMCA = input_data['S0_GMCA']
            sizeBlock = input_data['batch_size'][0,0].astype(int)
            n_s = input_data['n_s'][0,0]
            WNFactors = input_data['WNFactors'][0]
            X0_GMCA = input_data['X0']
            normOpt = input_data['normOpt']

            print('*******************************************')
            print(data_str)
            print('*******************************************')
            print('totalSize: ' + str(X0_GMCA.shape[1]*J))
            # print('divisors: ' + str(divisors))
            print('sizeBlock: ' + str(sizeBlock))
            # print('lineBlockSize: ' + str(lineBlockSize))
            print('numIts: ' + str(numIts))
            print('rho: ' + str(rho))
            print('alpha_init: ' + str(alpha_init))
            print('n_s: ' + str(n_s))
            print('n_obs: ' + str(X0_GMCA.shape[0]))
            print('J: ' + str(J))
            print('decF: ' + str(decF))
            print('normOpt_test: ' + str(normOpt_test))
            print('opt_GLOB_id_obs: ' + str(opt_GLOB_id_obs))
            print('***   ***   ***   ***   ***   ***   ***   ***')
            print('optGMCA: ' + str(optGMCA))
            # print('optBGMCA: ' + str(optBGMCA))
            print('optDecBGMCA: ' + str(optDecBGMCA))
            # print('optDecBGMCA2: ' + str(optDecBGMCA2))
            print('optDecBLGMCA: ' + str(optDecBLGMCA))
            print('*******************************************')


            X_GMCA, line_id2, r_id2, ccS2 = reshapeDecData(X,batch_size=sizeBlock,J=J)

            title_str = "Test_realDecData" + '_J_' + str(J) +  "_totalSize_" + str(S0.shape[1]) + "_sizeBlock_" + str(sizeBlock) + \
                "_n_obs_" + str(A0.shape[0]) + "_n_s_" + str(n_s) 

        else:

            X,X0,A0,S0,N = Make_Experiment_GG(n_s=n_s,n_obs=n_obs,t_samp=totalSize,noise_level=60.0,\
                                              dynamic=0,CondNumber=1,alpha=rho)
            X_GMCA = X
        
        if optGMCA == 1:
            Results_sB_totSC = gmca_original_SC(X_GMCA,n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,\
                        Aposit=False,BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,threshOpt=1\
                        ,SCOpt=1)
            A_sB_totSC = Results_sB_totSC['mixmat']
            S_sB_totSC = Results_sB_totSC['sources']
            crit_sB_totSC = EvalCriterion_fast(A0,S0_GMCA,A_sB_totSC,S_sB_totSC) 
            ca_final_GMCA[0,it_div,it_n] = crit_sB_totSC['ca_mean']
            ca_final_GMCA[1,it_div,it_n] = crit_sB_totSC['ca_med']
            if numIts == 1:
                print('-10*np.log10(ca_final_GMCA[:,%d,%d]): '%(it_div,it_n))
                print(-10*np.log10(ca_final_GMCA[:,it_div,it_n]))

        if optDecBGMCA == 1:
            Results_sB1 = bgmca3(X,n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
                                BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
                                threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=1,optA=1,alpha_exp=alpha_init,J=J,WNFactors=WNFactors,\
                                normOpt=normOpt)
            A_sB1 = Results_sB1['mixmat']
            S_sB1 = Results_sB1['sources']
            crit_sB1 = EvalCriterion_fast(A0,S0_GMCA,A_sB1,S_sB1) 
            ca_final_DGMCA[0,it_div,it_n] = crit_sB1['ca_mean']
            ca_final_DGMCA[1,it_div,it_n] = crit_sB1['ca_med']
            S_2D_sB1 = Results_sB1['images']
            if numIts == 1:
                print('-10*np.log10(ca_final_DGMCA[:,%d,%d]): '%(it_div,it_n))
                print(-10*np.log10(ca_final_DGMCA[:,it_div,it_n]))

        if optDecBLGMCA == 1:

            for it_ln in range(len(line_divs)):
                lineBlockSize = line_divs[it_ln]
                print('lineBlockSize: ' + str(lineBlockSize))

                Results_sB3 = blgmca4(X,n=n_s,maxts = 7,mints=3,nmax=100,L0=1,UseP=1,verb=0,Init=0,Aposit=False,\
                                    BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=1.,AInit=None,tol=1e-6,subBlockSize=sizeBlock,\
                                    threshOpt=2,weightFMOpt=1,SCOpt=1,alphaEstOpt=1,optA=1,alpha_exp=alpha_init,J=J,WNFactors=WNFactors,\
                                    normOpt=normOpt,lineBlockSize=lineBlockSize)
                A_sB3 = Results_sB3['mixmat']
                S_sB3 = Results_sB3['sources']
                crit_sB3 = EvalCriterion_fast(A0,S0_GMCA,A_sB3,S_sB3) 
                ca_final_xDGMCA[0,it_div,it_ln,it_n] = crit_sB3['ca_mean']
                ca_final_xDGMCA[1,it_div,it_ln,it_n] = crit_sB3['ca_med']
                S_2D_sB3 = Results_sB3['images']
                if numIts == 1:
                    print('-10*np.log10(ca_final_xDGMCA[:,%d,%d,%d]): '%(it_div,it_ln,it_n))
                    print(-10*np.log10(ca_final_xDGMCA[:,it_div,it_ln,it_n]))

print('*******************************************')
print('Finish')
print('*******************************************')
print('-10*np.log10(ca_final_GMCA): ')
print(-10*np.log10(ca_final_GMCA))
print('-10*np.log10(ca_final_DGMCA): ')
print(-10*np.log10(ca_final_DGMCA))
print('-10*np.log10(ca_final_xDGMCA): ')
print(-10*np.log10(ca_final_xDGMCA))
print('*******************************************')

title_str = "_v2_Test_transf_dataReport_" + '_J_' + str(J) + '_decF_' + str(decF) +  "_totalSize_" + str(S0.shape[1]) \
+ "_sizeBlock_" + str(len(divisors)) + "_n_obs_" + str(A0.shape[0]) 

if len(line_divs) == 1:
    title_str = title_str + "_line_divs_" + str(line_divs[0])
else:
    title_str = title_str + "_line_divs_" + str(len(line_divs))

# data_str = 'Ver6_lBsize25_' + data_str

if optGMCA == 1:
    print('** GMCA **')
    print('-10*np.log10(ca_final_GMCA): ')
    print(-10*np.log10(ca_final_GMCA))
    title_str_GMCA = save_data_path +  'ca_final_GMCA' + title_str 
    np.save(title_str_GMCA,ca_final_GMCA)
if optDecBGMCA == 1:
    print('** DGMCA **')
    print('-10*np.log10(ca_final_DGMCA): ')
    print(-10*np.log10(ca_final_DGMCA))
    title_str_DGMCA = save_data_path +  'ca_final_DGMCA' + title_str 
    np.save(title_str_DGMCA,ca_final_DGMCA)
if optDecBLGMCA == 1:
    print('** xDGMCA **')
    print('-10*np.log10(ca_final_xDGMCA): ')
    print(-10*np.log10(ca_final_xDGMCA))
    title_str_xDGMCA = save_data_path +  'ca_final_xDGMCA' + title_str 
    np.save(title_str_xDGMCA,ca_final_xDGMCA)
print('*******************************************')





# if J > 0:
#     title_str = "Test_realDecData" + '_J_' + str(J) +  "_totalSize_" + str(S0.shape[1]) + "_sizeBlock_" + str(sizeBlock) + \
#     "_n_obs_" + str(A0.shape[0]) + "_n_s_" + str(n_s)

#     title_ca_final_sB1 = data_path + 'results' + data_str + '_'  + title_str
#     results = {'A': A_sB1, 'S': S_sB1, 'S_2Ds':S_2D_sB1, 'A_GMCA': A_sB_totSC, 'S_GMCA': S_sB_totSC, 'X_GMCA': X_GMCA, 'A0': A0, 'S0': S0, 'X': X, \
#     'sizeBlock': sizeBlock, 'n_s': n_s, 'WNFactors': WNFactors, 'X0': X0_GMCA, 'perf_sB1': (-10*np.log10(ca_final_sB1))}
#     spio.savemat(title_ca_final_sB1,results)
# else:
#     title_str = "Test_realDecData" + "_totalSize_" + str(totalSize) + "_numDivisors_" + str(len(divisors))+ "_n_obs_" + str(n_obs) \
#         + "_n_s_" + str(n_s)




# Save variables

# title_ca_final_sB_totSC = data_path + 'ca_final_sB_totSC' + title_str  
# np.save(title_ca_final_sB_totSC,ca_final_sB_totSC)


# ca_final_sB_totSC_mean = np.mean(ca_final_sB_totSC,axis=1)
# ca_final_sB_totSC_med = np.median(ca_final_sB_totSC,axis=1)
# log_ca_final_sB_totSC_mean = -10*np.log10(ca_final_sB_totSC_mean)
# log_ca_final_sB_totSC_med = -10*np.log10(ca_final_sB_totSC_med)

# ca_final_sB1_mean = np.mean(ca_final_sB1,axis=2)
# ca_final_sB1_med = np.median(ca_final_sB1,axis=2)
# log_ca_final_sB1_mean = -10*np.log10(ca_final_sB1_mean)
# log_ca_final_sB1_med = -10*np.log10(ca_final_sB1_med)





################# Ploting script
# import matplotlib
# matplotlib.rcParams.update({'font.size': 18})
# fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
# plt.rc('text', usetex=True)

# plt.plot(divisors, log_ca_final_sB_totSC_med[0]*np.ones(len(divisors)), '-.x', label=r'$CA_{mean}(gmca)$')
# plt.plot(divisors, log_ca_final_sB_totSC_med[1]*np.ones(len(divisors)), '-.x', label=r'$CA_{med}(gmca)$')

# plt.plot(divisors, log_ca_final_sB1_med[0,:], '--o', label=r'$CA_{mean}(bgmca)$')
# plt.plot(divisors, log_ca_final_sB1_med[1,:], '-o', label=r'$CA_{med}(bgmca)$')


# plt.xlabel('Batch size')
# plt.ylabel(r'CA performance $\left(-10 \log_{10}(CA)\right)$')
# plt.title(r'Test basic. $Reg \ell_0$. w$SC$. SNR=$60$dB. $\rho = 0.5$. $\alpha_{exp}=2$.$numIts=' \
#     + str(numIts) + r'$ $n_s = ' + str(n_s) + r'$ $ n_obs = ' + str(n_obs) + r'$. $lineBS = ' \
#     + str(lineBlockSize) + r'$.' )
# plt.grid()
# plt.legend()
# plt.show()

# title_str_pdf = plots_path + title_str + ".pdf"

# fig.savefig(title_str_pdf, bbox_inches='tight')