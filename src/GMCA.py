"""
Original GMCA algorithm with the norm Inf thresholding strategy with AMCA reweighting option


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
            # 1 --> Threshold using exponential decay and norm_inf
    SCOpt : Weighting for the partially correlated sources
            # 0 --> Deactivated
            # 1 --> Activated

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

"""


from utils_dgmca import randperm
from utils_dgmca import mad
# Import from utils_dgmca -> randperm et mad

################# AMCA Main function

def GMCA(X,n=2,maxts = 7,mints=3,nmax=100,q_f = 0.1,L0=0,UseP=1,verb=0,Init=0,Aposit=False,BlockSize= None,\
    NoiseStd=[],IndNoise=[],Kmax=0.5,AInit=None,tol=1e-6,threshOpt=0,SCOpt=0):

    import numpy as np
    import scipy.linalg as lng
    import copy as cp

    nX = np.shape(X);
    m = nX[0];t = nX[1]
    Xw = cp.copy(X)

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

    S,A = Core_GMCA(X=Xw,A=A,S=S,n=n,maxts=maxts,q_f=q_f,BlockSize = BlockSize,Aposit=Aposit,tol=tol,kend = mints,\
        nmax=nmax,L0=L0,UseP=UseP,verb=verb,IndNoise=IndNoise,Kmax=Kmax,NoiseStd=NoiseStd,\
        threshOpt=threshOpt,SCOpt=SCOpt)

    Results = {"sources":S,"mixmat":A}

    return Results

################# AMCA internal code (Weighting the sources only)

def Core_GMCA(X=0,n=0,A=0,S=0,q_f = 0.1,maxts=7,kend=3,nmax=100,BlockSize = 2,L0=1,Aposit=False,UseP=0,verb=0,\
    IndNoise=[],Kmax=0.5,NoiseStd=[],tol=1e-6,threshOpt=0,SCOpt=0):

#--- Import useful modules
    import numpy as np
    import scipy.linalg as lng
    import copy as cp
    import scipy.io as sio
    import time

#--- Init

    n_X = np.shape(X)
    n_S = np.shape(S)

    k = maxts
    dk = (k-kend)/(nmax-1);
    perc = Kmax/nmax
    Aold = cp.deepcopy(A)

    alpha_exp = 0.5 #2. #0.05
    # sigma_noise = np.zeros([n])
    # for r in range(n):
    #     sigma_noise[r] = mad(S[r,:]) # Suposed to be known  
    K = kend
#-- AMCA parameters    
    alpha = 1;
    dalpha = (alpha-q_f)/nmax;

#
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

    Resi = cp.deepcopy(X)

#--- Main loop
    while Go_On:
        it += 1

        if it == nmax:
            Go_On = 0

        #--- Estimate the sources

        sigA = np.sum(A*A,axis=0)
        indS = np.where(sigA > 0)[0]

        if np.size(indS) > 0:

            # Using blocks

            IndBatch = randperm(len(indS))  #---- mini-batch amongst available sources

            if BlockSize+1 < len(indS):
                indS = indS[IndBatch[0:BlockSize]]

            Resi = Resi + np.dot(A[:,indS],S[indS,:])   # Putting back the sources

            Ra = np.dot(A[:,indS].T,A[:,indS])
            Ua,Sa,Va = np.linalg.svd(Ra)
            Sa[Sa < np.max(Sa)*1e-9] = np.max(Sa)*1e-9
            iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
            piA = np.dot(iRa,A[:,indS].T)
            S[indS,:] = np.dot(piA,Resi)

            # Propagation of the noise statistics

            if len(NoiseStd) > 0:
                SnS = 1./(n_X[1]**2)*np.diag(np.dot(piA,np.dot(np.diag(NoiseStd**2),piA.T)))

#---------- Udpating the AMCA weights
            if SCOpt == 1:
                Su = np.dot(np.diag(1./np.maximum(1e-9,np.std(S,axis=1))),S) # Normalizing the sources
                wa = 1./n*np.power(np.sum(np.power(abs(Su[indS,:]),alpha),axis=0),1./alpha)
                Inz = np.where(wa > 1e-24)[0]
                Iz = np.where(wa < 1e-24)[0]
                wa = 1./(wa+1e-24);
                if len(Iz) > 0:
                    wa[Iz] = np.median(wa[Inz])
                wa = wa/np.max(wa)
                alpha -= dalpha

#---------- Thresholding
            Stemp = S[indS,:]
            Sref = cp.copy(S)

            for r in range(len(indS)):

                St = Stemp[r,:]

##------------- Opt 0
                if threshOpt == 0:
                    if len(IndNoise) > 0:
                        tSt = kend*mad(St[IndNoise])
                    elif len(NoiseStd) > 0:
                        tSt = kend*np.sqrt(SnS[indS[r]])
                    else:
                        tSt = kend*mad(St)

                    indNZ = np.where(abs(St) - tSt > 0)[0]
                    thrd = mad(St[indNZ])

                    if UseP == 0:
                        thrd = k*thrd

                    if UseP == 1:
                        Kval = np.min([np.floor(np.max([2./n_S[1],perc*it])*len(indNZ)),n_S[1]-1.])
                        I = abs(St[indNZ]).argsort()[::-1]
                        Kval = np.int(np.min([np.max([Kval,5.]),len(I)-1.]))
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

##------------- Opt 1
                elif threshOpt == 1:
                    S_norm_inf = np.amax(abs(St))*0.99
                    sigma_noise = abs(mad(St))
                    thrd = K*sigma_noise + (S_norm_inf - K*sigma_noise)*np.exp(-1.*(it-2)*alpha_exp)

                    St[(abs(St) < thrd)] = 0
                    indNZ = np.where(abs(St) >= thrd)[0]

                    if len(indNZ) == 0:
                        print(it)
                        print('len(indNZ) == 0')
                        print('Thrd: %f' %(thrd))
                        print('S_norm_inf: %f' %(S_norm_inf))
                        print('sigma_noise: %f' %(sigma_noise))
                        # St = np.zeros(np.shape(St)) # No value passed the thresholding

                    # if L0 == 0:
                    #     St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ])


                Stemp[r,:] = St

            S[indS,:] = Stemp

            k = k - dk

#------ Updating the mixing matrix
        Xr = cp.deepcopy(Resi)

        if SCOpt == 1:
            Sr = S*np.tile(wa,(n,1))
            Rs = np.dot(S[indS,:],Sr[indS,:].T) # It might be better to update non-zero sources

            Us,Ss,Vs = np.linalg.svd(Rs)
            Ss[Ss < np.max(Ss)*1e-9] = np.max(Ss)*1e-9
            iRs = np.dot(Us,np.dot(np.diag(1./Ss),Vs))
            piS = np.dot(Sr[indS,:].T,iRs)
            A[:,indS] = np.dot(Xr,piS)

        elif SCOpt == 0:
            Rs = np.dot(S[indS,:],S[indS,:].T)
            Us,Ss,Vs = np.linalg.svd(Rs)
            Ss[Ss < np.max(Ss)*1e-9] = np.max(Ss)*1e-9
            iRs = np.dot(Us,np.dot(np.diag(1./Ss),Vs))
            piS = np.dot(S[indS,:].T,iRs)
            A[:,indS] = np.dot(Resi,piS)


        if Aposit:
            for r in range(len(indS)):
                if A[(abs(A[:,indS[r]])==np.max(abs(A[:,indS[r]]))),indS[r]] < 0:
                    A[:,indS[r]] = -A[:,indS[r]]
                    S[indS[r],:] = -S[indS[r],:]
                A = A*(A>0)

        A = A/np.maximum(1e-24,np.sqrt(np.sum(A*A,axis=0)))

        DeltaA =  np.max(abs(1.-abs(np.sum(A*Aold,axis=0)))) # Angular variations

        if DeltaA < tol:
            if it > 500:
                Go_On = 0

        if verb:
            print("Iteration #",it," - Delta = ",DeltaA)

        Aold = cp.deepcopy(A)

        Resi = Resi - np.dot(A[:,indS],S[indS,:])  # Re-defining the residual

    if verb:
        elapsed_time = time.time() - start_time
        print("Stopped after ",it," iterations, in ",elapsed_time," seconds")
    #
    return S,A