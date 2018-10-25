"""
Distributed Generalized Morphological Analysis (DGMCA)

This code is not actually parallelized but it was done to allow an easy preparation of a 
future parallelized version.
The code solves the Blind Source Separation (BSS) problem giving as a result the 
source and the mixing matrices.

Solving de BSS problem when the images are sparse in another domain.
The parallelization is done only over the columns.
Permutation of the columns is optional via the shufflingOpt parameter (Recomended).
Threshold strategy: using exponential decay and norm_inf
Weight for FM strategy: SNR over the S space

  Usage:
    Results = GMCA(X,n=2,mints=3,nmax=100,L0=0,verb=0,Init=0,BlockSize= None,Kmax=0.5,AInit=None,tol=1e-6)
    Results = DGMCA(X,n=5,mints=3,nmax=100,L0=1,verb=0,Init=0,BlockSize= None,Kmax=1.,AInit=None,tol=1e-6,\
                    subBlockSize=500, SCOpt=1, alphaEstOpt=1,alpha_exp=2.)

  Inputs:
    X            : m x t array (input data, each row corresponds to a single observation)
    n            : scalar (number of sources to be estimated)
    mints        : scalar (final value of the k-mad thresholding)
    nmax         : scalar (number of iterations)
    q_f          : scalar (amca parameter to evaluate to correlation between the sources; should be lower than 1)
    L0           : if set to 1, L0 rather than L1 penalization
    verb         : boolean (if set to 1, in verbose mode)
    Init         : scalar (if set to 0: PCA-based initialization, if set to 1: random initialization)
    BlockSize    : scalar (should be lower than number of sources. The number of random sources to update at each iteration)
    Kmax         : scalar (maximal L0 norm of the sources. Being a percentage, it should be between 0 and 1)
    AInit        : if not None, provides an initial value for the mixing matrix
    tol          : scalar (tolerance on the mixing matrix criterion)
    subBlockSize : scalar (size of the subproblem (t_j), must be smaller than t(columns of X).)
    alpha_exp    : scalar (parameter controling the exponential decay of the thresholding strategy)
    alphaEstOpt  : Use online estimation for the alpha_exp thresholding parameter
                    # 0 --> Deactivated
                    # 1 --> Activated
    SCOpt        : Weighting for the partially correlated sources
                    # 0 --> Deactivated
                    # 1 --> Activated
    J            : scalar (wavelet decomposition level. J=0 means )
    WNFactors    :
    normOpt      :

  Outputs:
   Results : dict with entries:
        if J = 0:
            sources    : n x t array (estimated sources)
            mixmat     : m x n array (estimated mixing matrix)
        if J > 0:
            sources    : n x t array (estimated sources)
            mixmat     : m x n array (estimated mixing matrix)
            images     :
            
  Description:
    Computes the sources and the mixing matrix with GMCA

  Example:
     S,A = GMCA(X,n=2,mints=0,nmax=500) will perform GMCA assuming that the data are noiseless

  Version
    v1 - November,28 2017 - J.Bobin - CEA Saclay
    v2 - ... 2018 - T.Liaudat - CEA Saclay

"""

import numpy as np
import scipy.linalg as lng
import copy as cp
import time
from utils2 import randperm
from utils2 import mad
from FrechetMean import FrechetMean
from misc_bgmca2 import *
from starlet import *

################# DGMCA Main function

def DGMCA(X,n=2,mints=3,nmax=100,q_f=0.1,L0=0,verb=0,Init=0,BlockSize= None,\
    Kmax=0.5,AInit=None,tol=1e-6,subBlockSize=100,alphaEstOpt=1,SCOpt=1,alpha_exp=2.,J=0\
    ,WNFactors=0,normOpt=0):

    nX = np.shape(X);
    m = nX[0];t = nX[1]  

    if J > 0:
        Xw, line_id, r_id, ccX = reshapeDecData(X,batch_size=subBlockSize,J=J,normOpt=normOpt,WNFactors=WNFactors)
        print Xw.shape
    else:
        line_id = 0
        r_id = 0
        Xw = X 

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
    S,A = Core_DGMCA(X=Xw,A=A,S=S,n=n,q_f=q_f,BlockSize = BlockSize,tol=tol,kend = mints,nmax=nmax,L0=L0,\
        verb=verb,Kmax=Kmax,subBlockSize=subBlockSize,SCOpt=SCOpt,alphaEstOpt=alphaEstOpt,alpha_exp=alpha_exp);

    if J > 0:
        Ra = np.dot(A.T,A)
        Ua,Sa,Va = np.linalg.svd(Ra)
        Sa[Sa < np.max(Sa)*1e-9] = np.max(Sa)*1e-9
        iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
        piA = np.dot(iRa,A.T)
        ccS = np.dot(piA,ccX)

        S_2Ds = recoverDecData(S,r_id,line_id,J=J,ccX=ccS,deNormOpt=normOpt,WNFactors=WNFactors)

        Results = {"sources":S,"mixmat":A,"images":S_2Ds}

    else:
        Results = {"sources":S,"mixmat":A}


    return Results

################# DGMCA internal code

def Core_DGMCA(X=0,n=0,A=0,S=0,kend=3,q_f=0.1,nmax=100,BlockSize = 2,L0=1,verb=0,Kmax=0.5,\
    tol=1e-6, subBlockSize=100, SCOpt=1,alphaEstOpt=1,alpha_exp=2.):

#--- Import useful modules
    # import numpy as np
    # import scipy.linalg as lng
    # import copy as cp
    # import time

#--- Initialization variables
    shufflingOpt = 1 # Complete shuffle of the columns (pixels)

    perc = Kmax/(nmax*1.) # Kmax should be 1
    Aold = cp.deepcopy(A)

    if SCOpt == 1: 
        alpha = 1
        dalpha = (alpha-q_f)/nmax

    K = kend # K from the K-sigma_MAD strategy for the threshold final value

    # Batch related variables
    numBlocks = np.ceil(X.shape[1]/(subBlockSize*1.)).astype(int) # Number of blocks
    lastBlockSize = X.shape[1]%subBlockSize # Size of the last block if its not the same size as the other blocks

    # FM related variables
    A_block = np.zeros([A.shape[0],A.shape[1],numBlocks])
    A_mean = A
    w_FM = np.zeros([n,numBlocks]) # FM weight of each column of each estimation of A

    # Threshold related variables
    thresh_block = np.zeros([n,numBlocks]) 
    mad_block = np.zeros([n,numBlocks]) 

    # alpha_r parameter estimation
    GG_maxIt = 50
    GG_warm_start = 0
    if alphaEstOpt == 1:
        GG_warm_start = 1
        A_init = cp.deepcopy(A)

#
    Go_On = 1
    it = 1
#
    if verb:
        print("Starting main loop ...")
        print(" ")
        print("  - Final k: ",kend)
        print("  - Maximum number of iterations: ",nmax)
        if L0:
            print("  - Using L0 norm rather than L1")
        print(" ")
        print(" ... processing ...")
        start_time = time.time()


#--- Main loop
    while Go_On:

        it += 1

        if it == nmax:
            Go_On = 0
        
        # print('it: %d / %d'%(it,nmax))

#------ Estimation of the GG parameters
        if it == GG_maxIt and GG_warm_start==1:
            alpha_exp = alpha_thresh_estimation(A_mean, X, alpha_exp,n)
            A_mean = A_init # Restart from the initialization point
            S = np.dot(A_mean.T,X)

            GG_warm_start = 0
            it = 2 # Restart the main loop
            if SCOpt == 1: 
                alpha = 1
                dalpha = (alpha-q_f)/nmax


#------ Initializaion of the blocks
        indBBlocks = randperm(numBlocks) # Shuffling the blocks indexes
        valB = np.arange(numBlocks)*subBlockSize

        if lastBlockSize == 0:
            valB = np.append(valB,numBlocks*subBlockSize)
        else:
            valB = np.append(valB,(numBlocks-1)*subBlockSize + lastBlockSize)

        # Save a copy of the sources before modifications
        Sref = cp.deepcopy(S) 

#------ Shuffling the columns
        if shufflingOpt == 1: 
            if it == 2:
                original_id = np.arange(S.shape[1])

            shuffling_id = randperm(S.shape[1])
            original_id = original_id[shuffling_id] # Keep track of the permutations
            S = S[:,shuffling_id]
            X = X[:,shuffling_id]

#------ Thresholding calculs 
        threshold = np.zeros([n])
        S_norm_inf = np.amax(abs(thresh_block),axis=1) # Calculation of the total maxima (max of maxs)
        mad_S = np.median(mad_block,axis=1) # Naive distributed estimation of mad(S)

#------ Warm up rounds substitute
        if it == 2:
            S_norm_inf = np.amax(abs(S),axis=1)
            for itB in range(numBlocks):
                for r in range(n):
                    mad_block[r,itB] = mad(S[r,valB[itB]:valB[itB+1]])
            mad_S = np.median(mad_block,axis=1)             

#------ Threshold calculation
        for r in range(n):
            # sigma_noise = mad(S[r,:]) # Suposed to be known
            sigma_noise = mad_S[r]                       
            threshold[r] = K*sigma_noise + (S_norm_inf[r] - K*sigma_noise)*np.exp(-1*(it-2)*alpha_exp)

#-------------------------------------
##----- Block loop
        for itB in range(numBlocks):

            # print('itB: %d / %d'%(itB,numBlocks))
            A = cp.deepcopy(A_mean) 

            sigA = np.sum(A*A,axis=0)
            indS = np.where(sigA > 0)[0]

            if np.size(indS) > 0:

                iB = indBBlocks[itB] # Batch to be treated

                # Using blocks (in the GMCA way)
                IndBatch = randperm(len(indS))  # Mini-batch amongst available sources
                if BlockSize+1 < len(indS):
                    indS = indS[IndBatch[0:BlockSize]]

#-------------- Estimate the sources
                Ra = np.dot(A[:,indS].T,A[:,indS])
                Ua,Sa,Va = np.linalg.svd(Ra)
                Sa[Sa < np.max(Sa)*1e-9] = np.max(Sa)*1e-9
                iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
                piA = np.dot(iRa,A[:,indS].T)
                S[indS,valB[iB]:valB[iB+1]] = np.dot(piA,X[:,valB[iB]:valB[iB+1]])


#-------------- Udpating the AMCA block weights
                if SCOpt == 1:
                    S_up = S[indS,valB[iB]:valB[iB+1]]
                    Su = np.dot(np.diag(1./np.maximum(1e-9,np.std(S_up,axis=1))),S_up) # Normalizing the sources
                    wa = 1./n*np.power(np.sum(np.power(abs(Su[indS,:]),alpha),axis=0),1./alpha)
                    Inz = np.where(wa > 1e-24)[0]
                    Iz = np.where(wa < 1e-24)[0]
                    wa = 1./(wa+1e-24);
                    if len(Iz) > 0:
                        wa[Iz] = np.median(wa[Inz])
                    wa = wa/np.max(wa)


#-------------- Thresholding
                Stemp = S[indS,valB[iB]:valB[iB+1]]
  
                src_nnz = np.ones(len(indS),dtype=bool) # Used to identify non zero estimations

                sigma_X_j = mad(X[:,valB[iB]:valB[iB+1]])*1. # For the weighting strategy of the FM
                piA_normF2 = np.sum(np.diag(np.dot(piA.T,piA))) # For the weighting strategy of the FM

        #------ Loop over the sources
                for r in indS:
                    St = Stemp[r,:]

            #------ Retrieving statistics
                    thresh_block[r,iB] = np.max(abs(St))*0.99 # Max calcul in each 'cluster'
                    mad_block[r,iB] = mad(St)
                    thrd = threshold[r]

            #------ Thresholding a source
                    St[(abs(St) < thrd)] = 0
                    indNZ = np.where(abs(St) >= thrd)[0]

                    if len(indNZ) == 0:
                        St = np.zeros(np.shape(St)) # No value passed the thresholding
                        src_nnz[r] = False # Zero estimation identified

                    if L0 == 0:
                        St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ])

                    Stemp[r,:] = St

            #------ Weight calculation for the FM
                    w_FM[r,iB] = np.sum(St**2)/((len(St)*1.)*(sigma_X_j*piA_normF2))

        #------ End of the sources loop
                S[indS[src_nnz],valB[iB]:valB[iB+1]] = Stemp[indS[src_nnz],:]


# --------- Updating the mixing matrix with or without the AMCA weights
            if np.any(src_nnz):
                if SCOpt == 1:
                    Sr = S[indS[src_nnz],valB[iB]:valB[iB+1]]*np.tile(wa,(len(indS[src_nnz]),1))
                    Rs = np.dot(S[indS[src_nnz],valB[iB]:valB[iB+1]],Sr.T)
                    Us,Ss,Vs = np.linalg.svd(Rs)
                    Ss[Ss < np.max(Ss)*1e-9] = np.max(Ss)*1e-9
                    iRs = np.dot(Us,np.dot(np.diag(1./Ss),Vs))
                    piS = np.dot(Sr.T,iRs)
                    A[:,indS[src_nnz]] = np.dot(X[:,valB[iB]:valB[iB+1]],piS)

                elif SCOpt == 0:
                    Rs = np.dot(S[indS[src_nnz],valB[iB]:valB[iB+1]],S[indS[src_nnz],valB[iB]:valB[iB+1]].T)
                    Us,Ss,Vs = np.linalg.svd(Rs)
                    Ss[Ss < np.max(Ss)*1e-9] = np.max(Ss)*1e-9
                    iRs = np.dot(Us,np.dot(np.diag(1./Ss),Vs))
                    piS = np.dot(S[indS[src_nnz],valB[iB]:valB[iB+1]].T,iRs)
                    A[:,indS[src_nnz]] = np.dot(X[:,valB[iB]:valB[iB+1]],piS)

                # Normalization
                A[:,indS[src_nnz]] = A[:,indS[src_nnz]]/np.maximum(1e-24,np.sqrt(np.sum(A[:,indS[src_nnz]]*A[:,indS[src_nnz]],axis=0)))

                DeltaA =  np.max(abs(1.-abs(np.sum(A[:,src_nnz]*Aold[:,src_nnz],axis=0)))) # Angular variations
                if DeltaA < tol:
                    if it > 500:
                        Go_On = 0


            A_block[:,indS[src_nnz],iB] = A[:,indS[src_nnz]] # Block update for A

#------ Fusion of mixing matrices A
        if numBlocks == 1:
            A_mean = A
        else:

#---------- Normalization
            nnz_cols = (np.sum(abs(w_FM),axis=1) > 0)
            # Normalize weights (only where the lines dont sum zero)
            w_FM[nnz_cols,:] = np.divide(w_FM[nnz_cols,:],\
                np.reshape(np.sum(abs(w_FM[nnz_cols,:]),axis=1)*1.,((np.sum(abs(w_FM[nnz_cols,:]),axis=1)*1.).shape[0],1)))

#---------- Fusion method: The Frechet Mean
            A_mean = DGMCA_FrechetMean(As = A_block, Aref = A_mean, w = w_FM)


        Aold = cp.deepcopy(A_mean)

#------ AMCA matrix parameter update
        if SCOpt == 1:
            alpha -= dalpha 

#------ End of the block loop
        if verb:
            print("Iteration #",it," - Delta = ",DeltaA)

#--- End of the main loop
    if verb:
        elapsed_time = time.time() - start_time
        print("Stopped after ",it," iterations, in ",elapsed_time," seconds")

#-- Putting the sources back in order
    if shufflingOpt == 1:
        S = S[:,original_id]

# Goodbye
    return S,A_mean
