"""
Good estimation of wa version 2 (good one)

WORKING!

Solving de BSS problem when the images are sparse in another domain.
The parallelization is done  over the columns AND the lines.


  Usage:
    Results = GMCA(X,n=2,maxts = 7,mints=3,nmax=100,L0=0,UseP=1,verb=0,Init=0,Aposit=False,BlockSize= None,NoiseStd=[],IndNoise=[],Kmax=0.5,AInit=None,tol=1e-6)


  Inputs:
    X    : m x t array (input data, each row corresponds to a single observation)
    n     : scalar (number of sources to be estimated)
    maxts : scalar (initial value of the k-mad thresholding)
    mints : scalar (final value of the k-mad thresholding)
    L0 : if set to 1, L0 rather than L1 penalization
    nmax  : scalar (number of iterations)
    UseP  : boolean (if set to 1, thresholds as selected based on the selected source support: a fixed number of active entries is selected at each iteration)
    Init  : scalar (if set to 0: PCA-based initialization, if set to 1: random initialization)
    Aposit : if True, imposes non-negativity on the mixing matrix
    verb   : boolean (if set to 1, in verbose mode)
    NoiseStd : m x 1 array (noise standard deviation per observations)
    IndNoise : array (indicates the indices of the columns of S from which the k-mad is applied)
    Kmax : scalar (maximal L0 norm of the sources. Being a percentage, it should be between 0 and 1)
    AInit : if not None, provides an initial value for the mixing matrix
    tol : tolerance on the mixing matrix criterion
    threshOpt : Thresholding option
                # 0 --> Threshold using MAD of each block and the relaxing percentage of taken values
                # 1 --> Threshold using median of MADs of each block (using the relaxing percentage of taken values) 
                # 2 --> Threshold using exponential decay and norm_inf
    weightFMOpt : Option for the weighting strategy in the FM
                # 0 --> SNR over the X space
                # 1 --> SNR over the S space
                # 2 --> regular weighting (w=1/n)
    SCOpt : Weighting for the partially correlated sources
                # 0 --> Deactivated
                # 1 --> Activated
    alphaEstOpt : Use online estimation of alpha_exp thresholding parameter (for weightFMOpt=2)
                # 0 --> Deactivated
                # 1 --> Activated
    optA : Using the initial (or previously estimated) values of A and S after the estimation of the alpha (for alphaEstOpt=1)
                # 0 --> New values
                # 1 --> Old values 

  Outputs:
   Results : dict with entries:
        sources    : n x t array (estimated sources)
        mixmat     : m x n array (estimated mixing matrix)

  Description:
    Computes the sources and the mixing matrix with GMCA

  Example:
     S,A = GMCA(X,n=2,mints=0,nmax=500,UseP=1) will perform GMCA assuming that the data are noiseless

  Version
    v1 - November,28 2017 - J.Bobin - CEA Saclay
    v2 - ... 2018 - T.Liaudat - CEA Saclay

"""

from utils2 import randperm
from utils2 import mad
from FrechetMean import FrechetMean
from misc_bgmca2 import *
from starlet import *


################# AMCA Main function

def BLGMCA(X,n=2,maxts = 7,mints=3,nmax=100,q_f=0.1,L0=0,UseP=1,verb=0,Init=0,Aposit=False,BlockSize= None,NoiseStd=[],\
    IndNoise=[],Kmax=0.5,AInit=None,tol=1e-6,subBlockSize=100,lineBlockSize=1,threshOpt=1,weightFMOpt=1,alphaEstOpt=1,\
    optA =0,SCOpt=1,alpha_exp=2.,J=0 ,WNFactors=0,normOpt=0):

    import numpy as np
    import scipy.linalg as lng
    import copy as cp

    nX = np.shape(X);
    m = nX[0];t = nX[1]

    if J > 0:
        Xw, line_id, r_id, ccX = reshapeDecData(X,batch_size=subBlockSize,J=J,normOpt=normOpt,WNFactors=WNFactors)
        print Xw.shape
    else:
        line_id = 0
        r_id = 0
        Xw = X # Xw = cp.copy(X)

    if BlockSize == None:
        BlockSize = n

    if verb:
        print("Initializing ...")
    if Init == 0:
        R = np.dot(Xw,Xw.T)
        D,V = lng.eig(R)
        A = V[:,0:n]
    if Init == 1:
        A = np.random.randn(m,n)
    if AInit != None:
        A = cp.deepcopy(AInit)

    for r in range(0,n):
        A[:,r] = A[:,r]/lng.norm(A[:,r]) # - We should do better than that

    S = np.dot(A.T,Xw);

    # Call the core algorithm

    S,A = Core_GMCA(X=Xw,A=A,S=S,n=n,maxts=maxts,q_f=q_f,BlockSize = BlockSize,Aposit=Aposit,tol=tol,kend = mints,nmax=nmax,L0=L0,\
        UseP=UseP,verb=verb,IndNoise=IndNoise,Kmax=Kmax,NoiseStd=NoiseStd,subBlockSize=subBlockSize,lineBlockSize=lineBlockSize,\
        threshOpt=threshOpt,weightFMOpt=weightFMOpt,SCOpt=SCOpt,optA=optA,alphaEstOpt=alphaEstOpt,alpha_exp=alpha_exp);

    if J > 0: 
        Ra = np.dot(A.T,A)
        Ua,Sa,Va = np.linalg.svd(Ra)
        Sa[Sa < np.max(Sa)*1e-9] = np.max(Sa)*1e-9
        iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
        piA = np.dot(iRa,A.T)
        ccS = np.dot(piA,ccX)

        S_2Ds = recoverDecData(S,r_id,line_id,J=J,ccX=ccS)

        Results = {"sources":S,"mixmat":A,"images":S_2Ds}
        
    else:
        Results = {"sources":S,"mixmat":A}


    return Results





################# AMCA internal code (Weighting the sources only)

def Core_GMCA(X=0,n=0,A=0,S=0,maxts=7,kend=3,q_f=0.1,nmax=100,BlockSize = 2,L0=1,Aposit=False,UseP=0,verb=0,IndNoise=[],Kmax=0.5,\
    NoiseStd=[],tol=1e-6, subBlockSize=100,lineBlockSize=1, threshOpt=1,weightFMOpt=0, SCOpt=1,optA=0,alphaEstOpt=1,alpha_exp=2.):

#--- Import useful modules
    import numpy as np
    import scipy.linalg as lng
    import copy as cp
    import scipy.io as sio
    import time

#--- Init

    shufflingOpt = 1

    n_X = np.shape(X)
    n_S = np.shape(S)

    k = maxts
    dk = (k-kend)/((nmax-1)*1.)
    perc = Kmax/(nmax*1.) # Kmax should be 1
    # Aold = cp.deepcopy(A)

    if SCOpt == 1: 
        alpha = 1
        dalpha = (alpha-q_f)/nmax

    K = kend # No need for the K variable if kend will be finally chosen

    numBlocks = np.ceil(X.shape[1]/(subBlockSize*1.)).astype(int) # Number of blocks
    lastBlockSize = X.shape[1]%subBlockSize # Size of the last block if its not the same size as the other blocks
    numLineBlocks = np.ceil(X.shape[0]/(lineBlockSize*1.)).astype(int) # Number of lineBlocks
    lastLineBlockSize = X.shape[0]%lineBlockSize

    # print('numBlocks=%d'%(numBlocks))
    # print('lastBlockSize=%d'%(lastBlockSize))
    # print('numLineBlocks=%d'%(numLineBlocks))
    # print('lastLineBlockSize=%d'%(lastLineBlockSize))

    A_block = np.zeros([A.shape[0],A.shape[1],numBlocks])
    A_mean = A

    S_block_std = np.zeros([n,subBlockSize,numLineBlocks])
    if lastBlockSize != 0:
        S_block_last = np.zeros([n,lastBlockSize,numLineBlocks])

    w_Ai = np.zeros([numBlocks]) # weight of each estimation of A (to be used with the Frechet Mean (FM))
    w_Sj = np.zeros([numLineBlocks]) # Weight of each eatimation of Sj from each line of A and a part of X
    w_FM = np.zeros([n,numBlocks]) # weight of each column of each estimation of A (to be used with the Frechet Mean (FM))

    sigma_noise_X = mad(X)*1. # The noise variance is supposed to be known

    thresh_block = np.zeros([n,numBlocks]) # Used for the calculation of the threshold
    historical_thrd = np.zeros([n,nmax-1])
    thresh_block_line = np.zeros([n,numBlocks,numLineBlocks])

    GG_maxIt = 50
    GG_warm_start = 0
    if alphaEstOpt == 1:
        # GG_mapping_param = np.array([ 4. , -7. ,  4.5])
        GG_mapping_param = np.array([ 4.5828 , -8.5807 ,  5.2610])
        GG_alpha_est = np.zeros([n])
        GG_warm_start = 1
        if optA == 1:
            A_init = cp.deepcopy(A)

##

    Go_On = 1
    it = 1
#
    if verb:
        print("Starting main loop ...")
        print(" ")
        print("  - Final k: ",kend)
        print("  - Maximum number of iterations: ",nmax)
        if UseP:
            print("  - Using support-based threshold estimation")
        if L0:
            print("  - Using L0 norm rather than L1")
        if Aposit:
            print("  - Positivity constraint on A")
        print(" ")
        print(" ... processing ...")
        start_time = time.time()

    if Aposit:
        A = abs(A)

#--- Main loop
    while Go_On:

        it += 1

        if it == nmax:
            Go_On = 0
        
        # print('it: %d / %d'%(it,nmax))

#-------------- Estimation of the GG parameters
        if it == GG_maxIt and GG_warm_start==1:
        #-- Source estimation from the mixing matrix estimation (A)
            Ra = np.dot(A_mean.T,A_mean)
            Ua,Sa,Va = np.linalg.svd(Ra)
            Sa[Sa < np.max(Sa)*1e-9] = np.max(Sa)*1e-9
            iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
            piA = np.dot(iRa,A_mean.T)
            S = np.dot(piA,X)

        #-- Rho estimation
            for r in range(n):
                S_sort = abs(cp.deepcopy(S[r,:]))
                S_sort.sort(axis=0)
                S_sort = S_sort[S_sort>0]

                if len(S_sort) != 0:
                    S_sort = S_sort/(np.sum(S_sort))
                    params = GG_parameter_estimation(S_sort,optHalf = 1)
                    GG_alpha_est[r] = params[0]
                else:
                    GG_alpha_est[r] = 0

        #-- Alpha_exp parameter estimation
            GG_alpha_est = GG_alpha_est[GG_alpha_est>0]
            GG_alpha_est = GG_alpha_est[~np.isnan(GG_alpha_est)]
            if len(GG_alpha_est) > 0:
                GG_alpha_est = np.median(GG_alpha_est) # Robust to outliers
            else:
                GG_alpha_est = alpha_exp # We say with the default value

        #-- Mapping the rho estimate with the alpha_exp value   
            print 'GG_rho_est: '
            print GG_alpha_est
            # Use of a quadratic model
            GG_alpha_est = GG_mapping_param[0]*(GG_alpha_est**2) + GG_mapping_param[1]*GG_alpha_est \
                            + GG_mapping_param[2]*np.ones(GG_alpha_est.shape)
            print 'GG_alpha_est: '
            print GG_alpha_est

        #-- Thresholding values o avoid divergences
            if GG_alpha_est > 20.:
                GG_alpha_est = 20.
            elif GG_alpha_est < 0.01:
                GG_alpha_est = 0.01

        #-- Update of parameters
            alpha_exp = GG_alpha_est
            GG_warm_start = 0
            it = 2
            # S = np.dot(A.T,Xw) # Check if this will help
            k = maxts
            dk = (k-kend)/((nmax-1)*1.)
            if SCOpt == 1: 
                alpha = 1
                dalpha = (alpha-q_f)/nmax
            if optA == 1:
                A_mean = A_init
                S = np.dot(A_mean.T,X)



#------ Initializaion of the columnBlocks
        indBBlocks = randperm(numBlocks) # Shuffling the blocks indexes
        valB = np.arange(numBlocks)*subBlockSize

        if lastBlockSize == 0:
            valB = np.append(valB,numBlocks*subBlockSize)
        else:
            valB = np.append(valB,(numBlocks-1)*subBlockSize + lastBlockSize)
        
#------ Initializaion of the lineBlocks       
        indLineBlocks = randperm(numLineBlocks) # Shuffling the line blocks indexes
        valLB = np.arange(numLineBlocks)*lineBlockSize
        if lastLineBlockSize == 0:
            valLB = np.append(valLB,numLineBlocks*lineBlockSize)
        else:
            valLB = np.append(valLB,(numLineBlocks-1)*lineBlockSize + lastLineBlockSize)


#------ Shuffling the columns
        if shufflingOpt == 1: 
            if it == 2:
                original_id = np.arange(S.shape[1])

            shuffling_id = randperm(S.shape[1])

            original_id = original_id[shuffling_id]
            S = S[:,shuffling_id]
            X = X[:,shuffling_id]


#------ Thresholding calculs (Restart gap)
        threshold = np.zeros([n])

        if threshOpt == 1:
            # Median calculation of the already calculed MADs
            thresh_block = np.median(thresh_block_line,axis=2)
            threshold = np.median(thresh_block,axis=1)

        elif threshOpt == 2:
            # Calculation of the total maxima (max of maxs)
            thresh_block = np.amax(abs(thresh_block_line),axis=2)
            S_norm_inf = np.amax(abs(thresh_block),axis=1)        


#------ Warm up rounds substitute
        if it == 2:
            if threshOpt == 1:
                for itB in range(numBlocks):
                    for r in range(n):
                        thresh_block[r,itB] = mad(S[r,valB[itB]:valB[itB+1]])

                threshold = np.median(thresh_block,axis=1)

            elif threshOpt == 2:
                S_norm_inf = np.amax(abs(S),axis=1)

            if weightFMOpt == 0:
                for itB in range(numBlocks):
                    weight = np.sum(np.diag(np.dot(X[:,valB[itB]:valB[itB+1]].T,X[:,valB[itB]:valB[itB+1]])))
                    w_Ai[itB] = weight/(sigma_noise_X*np.sqrt(valB[itB+1]-valB[itB]))
                w_Ai = w_Ai/(np.sum(w_Ai)*1.) # Normalize weights
                w_FM = np.ones([A_block.shape[1],A_block.shape[2]])
                w_FM = np.multiply(w_FM,w_Ai)
                print 'w_FM'
                print w_FM

            if weightFMOpt == 2:
                w_FM = np.ones([A_block.shape[1],A_block.shape[2]])
                w_FM = w_FM / (A_block.shape[1]*A_block.shape[2]*1.)
                # w_Ai = np.ones([numBlocks])
                # w_Ai = w_Ai/(np.sum(w_Ai)*1.) # Normalize weights


#------ Warm up rounds substitute
        if threshOpt == 2:
            for r in range(n):
                sigma_noise = mad(S[r,:]) # Suposed to be known                       
                threshold[r] = K*sigma_noise + (S_norm_inf[r] - K*sigma_noise)*np.exp(-1*(it-2)*alpha_exp)

                historical_thrd[r,it-2] = threshold[r]


#-------------------------------------
        sigA = np.sum(A_mean*A_mean,axis=0)
        indS = np.where(sigA > 0)[0]

#------ Calculation of the pseudo-inverse of each line of A_mean
        if lastLineBlockSize == 0:
            lastBlockLine = 0
            piA_L = np.zeros([len(indS),lineBlockSize,numLineBlocks])
            completeLineBlocks = numLineBlocks

        else:
            lastBlockLine = 1
            piA_L = np.zeros([len(indS),lineBlockSize,numLineBlocks-1])
            completeLineBlocks = numLineBlocks - 1
            piA_Llast = np.zeros([len(indS),lastLineBlockSize])

            itLlast = completeLineBlocks
            piA_Llast = pInv(A_mean[valLB[itLlast]:valLB[itLlast+1],indS])
            
        for itLB in range(completeLineBlocks):
            piA_L[:,:,itLB] = pInv(A_mean[valLB[itLB]:valLB[itLB+1],indS])

        piA_full = pInv(A_mean[:,indS])

#-------------------------------------
#------ Estimate the source

##----- Block loop
        for itB in range(numBlocks):

            A = cp.deepcopy(A_mean)
            # print('itB: %d / %d'%(itB,numBlocks))

            iB = indBBlocks[itB] # Block (batch) to be treated

            if iB == (numBlocks-1) and lastBlockSize != 0:
                S_block = S_block_last[indS,:,:]   
            else:
                S_block = S_block_std[indS,:,:]

            S_block_NonTh = cp.deepcopy(S_block)

            # # Initialize the weight
            # if weightFMOpt == 1:
            #     w_Ai[iB] = 0

##--------- Line loop
            for itLB in range(numLineBlocks):

                iL = indLineBlocks[itLB] # Block Line to be treated

                if lastBlockLine == 1 and iL == (numLineBlocks - 1):
                    piA = piA_Llast
                else:
                    piA = piA_L[:,:,iL]


                if np.size(indS) > 0:
                    # Using blocks
                    IndBatch = randperm(len(indS))  #---- mini-batch amongst available sources

                    if BlockSize+1 < len(indS):
                        indS = indS[IndBatch[0:BlockSize]]

                    Stemp = np.dot(piA,X[valLB[iL]:valLB[iL+1],valB[iB]:valB[iB+1]])

                    S_block_NonTh[:,:,iL] = cp.deepcopy(Stemp)

                    # Propagation of the noise statistics
                    if len(NoiseStd) > 0:
                        SnS = 1./(n_X[1]**2)*np.diag(np.dot(piA,np.dot(np.diag(NoiseStd**2),piA.T)))


# #------------------ Udpating the AMCA block weights
#                     if SCOpt == 1:
#                         Su = np.dot(np.diag(1./np.maximum(1e-9,np.std(Stemp,axis=1))),Stemp) # Normalizing the sources
#                         wa = 1./n*np.power(np.sum(np.power(abs(Su),alpha),axis=0),1./alpha)
#                         Inz = np.where(wa > 1e-24)[0]
#                         Iz = np.where(wa < 1e-24)[0]
#                         wa = 1./(wa+1e-24);
#                         if len(Iz) > 0:
#                             wa[Iz] = np.median(wa[Inz])
#                         wa = wa/np.max(wa)


#------------------ Thresholding
                    bad_estimation = 0        
                    sigma_X_j = mad(X[valLB[iL]:valLB[iL+1],valB[iB]:valB[iB+1]])*1.


                    for r in range(len(indS)):
                        St = Stemp[r,:]

#---------------------- Opt 0
                        if threshOpt == 0:
                            if len(IndNoise) > 0:
                                tSt = kend*mad(St[IndNoise])
                            elif len(NoiseStd) > 0:
                                tSt = kend*np.sqrt(SnS[indS[r]])
                            else:
                                tSt = kend*mad(St)

                            indNZ = np.where(abs(St) - tSt > 0)[0]
                            thrd = mad(St[indNZ])
                            if len(indNZ) == 0:
                                bad_estimation = 1
                                St = np.zeros(np.shape(St)) # No value passed the thresholding

                            if len(indNZ) > 0:
                                if UseP == 0:
                                    thrd = k*thrd

                                if UseP == 1:
                                    n_Stemp = np.shape(Stemp)
                                    Kval = np.min([np.floor(np.max([2./n_Stemp[1],perc*it/(1.*numBlocks)])*len(indNZ)),n_Stemp[1]-1.])
                                    I = abs(St[indNZ]).argsort()[::-1]
                                    Kval = np.int(np.min([np.max([Kval,n*1.]),len(I)-1.]))
                                    IndIX = np.int(indNZ[I[Kval]])
                                    thrd = abs(St[IndIX])

                                if UseP == 2:
                                    t_max = np.max(abs(St[indNZ]))
                                    t_min = np.min(abs(St[indNZ]))
                                    thrd = (0.5*t_max-t_min)*(1 - (it-1.)/(nmax-1)) + t_min # We should check that

                            St[(abs(St) < thrd)] = 0
                            indNZ = np.where(abs(St) > thrd)[0]

                            if L0 == 0:
                                St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ])

                            k = k - dk

#---------------------- Opt 1
                        if threshOpt == 1:
                            thresh_block_line[r,iB,iL] = mad(St)
                            # thresh_block[r,iB] = mad(St)
                            tSt = threshold[r]*kend
                            indNZ = np.where(abs(St) - tSt > 0)[0]

                            n_Stemp = np.shape(Stemp)
                            Kval = np.min([np.floor(np.max([2./n_Stemp[1],perc*it/(1.*numBlocks)])*len(indNZ)),n_Stemp[1]-1.])
                            I = abs(St[indNZ]).argsort()[::-1]

                            if len(I) != 0:
                                Kval = np.int(np.min([np.max([Kval,n*1.]),len(I)-1.]))
                                IndIX = np.int(indNZ[I[Kval]])
                                thrd = abs(St[IndIX])

                                St[(abs(St) < thrd)] = 0
                                indNZ = np.where(abs(St) > thrd)[0]

                                if L0 == 0:
                                    St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ])
                            else:
                                St = np.zeros(np.shape(St)) # No value passed the thresholding
                                
#---------------------- Opt 2
                        elif threshOpt == 2:
                            thresh_block_line[r,iB,iL] = np.max(abs(St))*0.99 # Max calcul in each 'cluster'
                            thrd = threshold[r]

                            St[(abs(St) < thrd)] = 0
                            indNZ = np.where(abs(St) > thrd)[0]

                            if len(indNZ) == 0:
                                St = np.zeros(np.shape(St)) # No value passed the thresholding

                            if L0 == 0:
                                St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ])

                        Stemp[r,:] = St


                    # weight = np.sum(np.diag(np.dot(X[valLB[iL]:valLB[iL+1]].T,X[valLB[iL]:valLB[iL+1]]))) # X space
                    weight = np.sum(np.diag(np.dot(Stemp.T,Stemp))) # S space (if Stemp is empty the weight will be 0)
                    w_Sj[iL] = weight / (sigma_X_j*np.sum(np.diag(np.dot(piA.T,piA))))
                    S_block[:,:,iL] = cp.deepcopy(Stemp)



#---------- Agregation of S estimators (easy solution: weighted average)
            if np.sum(abs(w_Sj)) != 0:
                w_Sj = w_Sj/(np.sum(abs(w_Sj))*1.) # sum = 1
            Stemp = np.sum(np.multiply(S_block,w_Sj),axis=2)
            Stemp_NonTh = np.sum(np.multiply(S_block_NonTh,w_Sj),axis=2)
            bad_estimation = 0

#------------------ Udpating the AMCA block weights
            if SCOpt == 1:
                Su = np.dot(np.diag(1./np.maximum(1e-9,np.std(Stemp_NonTh,axis=1))),Stemp_NonTh) # Normalizing the sources
                wa = 1./n*np.power(np.sum(np.power(abs(Su),alpha),axis=0),1./alpha)
                Inz = np.where(wa > 1e-24)[0]
                Iz = np.where(wa < 1e-24)[0]
                wa = 1./(wa+1e-24);
                if len(Iz) > 0:
                    wa[Iz] = np.median(wa[Inz])
                wa = wa/np.max(wa)

#------------------ Weight calcule (for the Frechet Mean)
                # if weightFMOpt == 1:
                #     weight = np.sum(np.diag(np.dot(Stemp.T,Stemp)))
                #     # w_Ai[iB] = weight/ (sigma_noise_X*np.sum(np.diag(np.dot(piA.T,piA))))
                #     w_Ai[iB] = weight / (sigma_noise_X*np.sum(np.diag(np.dot(piA_full.T,piA_full))))
            # else:
            #     Stemp = np.zeros([S_block.shape[0],S_block.shape[1]])
            #     bad_estimation = 1


            if weightFMOpt == 1:
                src_nnz = np.ones(len(indS),dtype=bool)
                for r in indS:
                    w_FM[r,iB] = np.sum(Stemp[r,:]**2)/((len(Stemp[r,:])*1.)*\
                        (sigma_X_j*np.sum(np.diag(np.dot(piA.T,piA)))))
                    if w_FM[r,iB] == 0:
                        src_nnz[r] = False

                if np.sum(src_nnz) == 0:
                    bad_estimation = 1

                # print('**********************')
                # print('it = %d, itB = %d, np.sum(src_nnz)=%d'%(it, itB, np.sum(src_nnz)))

#---------- Updating the mixing matrix with the AMCA weights
            # To escape from the case where a source in Stemp is completely zero
            if bad_estimation == 0:
                if SCOpt == 1:
                    Sr = Stemp[indS[src_nnz],:]*np.tile(wa,(len(indS[src_nnz]),1))
                    Rs = np.dot(Stemp[indS[src_nnz],:],Sr.T)
                    Us,Ss,Vs = np.linalg.svd(Rs) 
                    Ss[Ss < np.max(Ss)*1e-9] = np.max(Ss)*1e-9
                    iRs = np.dot(Us,np.dot(np.diag(1./Ss),Vs))
                    piS = np.dot(Sr.T,iRs)

                elif SCOpt == 0:
                    Rs = np.dot(Stemp[indS[src_nnz],:],Stemp[indS[src_nnz],:].T)
                    Us,Ss,Vs = np.linalg.svd(Rs)
                    Ss[Ss < np.max(Ss)*1e-9] = np.max(Ss)*1e-9
                    iRs = np.dot(Us,np.dot(np.diag(1./Ss),Vs))
                    piS = np.dot(Stemp[indS[src_nnz],:].T,iRs)

                for it_L in range(numLineBlocks):
                    A[valLB[it_L]:valLB[it_L+1],indS[src_nnz]] = np.dot(X[valLB[it_L]:valLB[it_L+1],valB[iB]:valB[iB+1]],piS)

                if Aposit:
                    for r in range(len(indS)):
                        if A[(abs(A[:,indS[r]])==np.max(abs(A[:,indS[r]]))),indS[r]] < 0:
                            A[:,indS[r]] = -A[:,indS[r]]
                            S[indS[r],:] = -S[indS[r],:]
                        A = A*(A>0)

                A = A/np.maximum(1e-24,np.sqrt(np.sum(A*A,axis=0)))

                DeltaA =  np.max(abs(1.-abs(np.sum(A*A_mean,axis=0)))) # Angular variations

                if DeltaA < tol:
                    if it > 500:
                        Go_On = 0


            elif bad_estimation == 1:
                w_Ai[iB] = 0
            

            A_block[:,:,iB] = A # Block update for A

#------ Fusion of mixing matrices A
        if numBlocks == 1:
            A_mean = A
        else:
            if weightFMOpt == 1:
                # if np.sum(w_Ai) != 0:

                # w_FM --> weight of each column of each estimation of A (to be used with the Frechet Mean (FM))
                w_nnz = np.sum(abs(A_block),axis=0)
                w_nnz[abs(w_nnz)>0] = np.ones(w_nnz[abs(w_nnz)>0].shape)
                w_FM = np.multiply(w_FM,w_nnz)

                nnz_cols = (np.sum(abs(w_FM),axis=1) > 0)
                # Normalize weights (only where the lines dont sum zero)
                w_FM[nnz_cols,:] = np.divide(w_FM[nnz_cols,:],\
                    np.reshape(np.sum(abs(w_FM[nnz_cols,:]),axis=1)*1.,((np.sum(abs(w_FM[nnz_cols,:]),axis=1)*1.).shape[0],1)))
                # The Frechet Mean function should manage the case where all the columns have weight zero and in that
                # case it should copy that column from the reference matrix (A_mean in this case)

                # w_Ai = w_Ai / (np.sum(abs(w_Ai))*1.)
            else:
                w_FM = np.zeros([A_block.shape[1],A_block.shape[2]])
                w_FM = np.ones(np.shape(w_FM)) / np.sum(np.ones(np.shape(w_FM)))

                    # w_Ai = np.ones(np.shape(w_Ai)) / np.sum(np.ones(np.shape(w_Ai)))


            # print 'A_block'
            # for it_tt in range(A_block.shape[2]):
            #     print('w_Ai[it_tt]=%f'%(w_Ai[it_tt]))
            #     print A_block[:,:,it_tt]
            # print 'w_FM'
            # print w_FM
            A_mean = easy_matrix_FrechetMean2(As = A_block, Aref = A_mean, w = w_FM)
            # print('it=%d, A_mean: '%(it))
            # print A_mean

        # Aold = cp.deepcopy(A_mean)

#------ Finish the block loop
        if verb:
            print("Iteration #",it," - Delta = ",DeltaA)

#------ AMCA matrix parameter update
        if SCOpt == 1:
            alpha -= dalpha 

#-- Finish the main loop

#-- Final S matrix estimation

    S = np.zeros([A_mean.shape[1],X.shape[1]])
    Api = pInv(A_mean)

    for it_f in range(numBlocks):
        S[:,valB[it_f]:valB[it_f+1]] = np.dot(Api,X[:,valB[it_f]:valB[it_f+1]])

    for r in range(n):
        sigma = mad(S[r,:])
        Sr = S[r,:]
        Sr[abs(Sr)<K*sigma] = 0
        S[r,:] = Sr


    if verb:
        elapsed_time = time.time() - start_time
        print("Stopped after ",it," iterations, in ",elapsed_time," seconds")

    # Putting the sources back in order
    if shufflingOpt == 1:
        S = S[:,original_id]

    return S,A_mean