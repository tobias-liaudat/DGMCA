# Miscellaneous functions for GMCA

###------------------------------------------------------------------------------###
 ################# Calculation of pseudo-inverse with threshold in the singular values

def pInv(A):
    import numpy as np

    min_cond = 1e-9
    Ra = np.dot(A.T,A)
    Ua,Sa,Va = np.linalg.svd(Ra)
    Sa[Sa < np.max(Sa)*min_cond] = np.max(Sa)*min_cond
    iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
    return np.dot(iRa,A.T)


###------------------------------------------------------------------------------###
 ################# Functions to estimate the Generalized Gaussian parameters
def GG_g_func(x,b,mu):
    import scipy as sp
    import numpy as np
    
    mu = mu *np.ones(x.shape)
    num = np.sum(((abs(x-mu))**b)*(np.log(abs(x-mu))))
    den = np.sum((abs(x-mu))**b) 
    
    return (1. + sp.special.polygamma(0,1/b)/b - num/den + np.log((b/len(x))*den)/b )


def GG_gprime_func(x,b,mu):
    import scipy as sp
    import numpy as np

    mu = mu *np.ones(x.shape)
    
    result = 1/(b**2) - sp.special.polygamma(0,1/b)/(b**2) - sp.special.polygamma(1,1/b)/(b**3)
    
    den = np.sum((abs(x-mu))**b)
    num1 = np.sum(((abs(x-mu))**b)*(np.log(abs(x-mu))))
    num2 = np.sum(((abs(x-mu))**b)*(np.log(abs(x-mu)))**2)
    
    return result - num2/den + (num1/den)**2 + num1/(b*den) - np.log((b/len(x))*den)/(b**2)


def GG_parameter_estimation(x,optHalf = 0):
    import numpy as np
    import scipy as sp
    
    maxIts = 100
    
    m1 = np.mean(x)
    m2 = np.std(x)  
    beta_0 = m1/m2
    
    if optHalf == 1:
    # If there are samples from one side of the distribution we asume the mean is zero
        m1 = 0.
        beta_0 = 1.0
    
    # beta estimation
    beta = beta_0
    for it in range(maxIts):
        beta = beta - 0.1*GG_g_func(x,beta,m1)/GG_gprime_func(x,beta,m1)

    # mu estimation
    mu = m1
    
    # alpha estimation
    alpha = ((beta/len(x))*np.sum((abs(x-mu))**beta))**(1/beta)
    
    return beta, alpha, mu

###------------------------------------------------------------------------------###
 ################# CALCULATE THE COLUMN PERMUTATION THAT WILL GIVE THE MAXIMUM TRACE

def maximizeTrace(mat):
#--- Import modules
    import numpy as np
    import copy as cp
    from munkres import Munkres
    
    assert mat.shape[0] == mat.shape[1] # Matrix should be square
#--- Use of the Kuhn-Munkres algorithm (or Hungarian Algorithm)    
    m = Munkres()
    # Convert the cost minimization problem into a profit maximization one
    costMatrix = np.ones(np.shape(mat))*np.amax(mat) - mat       
    indexes = m.compute(cp.deepcopy(costMatrix))
    
    indexes= np.array(indexes)
    
    return (indexes[:,1]).astype(int)

###------------------------------------------------------------------------------###
################# CODE TO COMPUTE THE MIXING MATRIX CRITERION (AND SOLVES THE PERMUTATION INDETERMINACY)

def CorrectPerm_fast(cA0,S0,cA,S,incomp=0):

    import numpy as np
    from scipy import special
    import scipy.linalg as lng
    import copy as cp
    from copy import deepcopy as dp
    import math

    A0 = cp.copy(cA0)
    A = cp.copy(cA)

    nX = np.shape(A0)

    IndOut = np.linspace(0,nX[1]-1,nX[1])
    IndE = np.linspace(0,nX[1]-1,nX[1])

    if incomp==0:

        for r in range(0,nX[1]):
            A[:,r] = A[:,r]/(1e-24+lng.norm(A[:,r]))
            A0[:,r] = A0[:,r]/(1e-24+lng.norm(A0[:,r]))

        # Diff = abs(np.dot(lng.inv(np.dot(A0.T,A0)),np.dot(A0.T,A)))

        try:
            Diff = abs(np.dot(lng.inv(np.dot(A0.T,A0)),np.dot(A0.T,A)))
        except np.linalg.LinAlgError:
            Diff = abs(np.dot(np.linalg.pinv(A0),A))
            print('WARNING, PSEUDO INVERSE TO CORRECT PERMUTATIONS')

        Sq = np.ones(np.shape(S))

        ind = maximizeTrace(Diff)

        Aq = A[:,ind]
        Sq = S[ind,:]

        A0q = A0
        S0q = S0

        IndE = ind.astype(int)

        for ns in range(0,nX[1]):
            p = np.sum(Sq[ns,:]*S0[ns,:])
            if p < 0:
                Sq[ns,:] = -Sq[ns,:]
                Aq[:,ns] = -Aq[:,ns]

    else:

        n = nX[1]
        n_e = np.shape(A)[1]   # Estimated number of sources

        for r in range(n):
            A0[:,r] = A0[:,r]/(1e-24+lng.norm(A0[:,r]))

        for r in range(n_e):
            A[:,r] = A[:,r]/(1e-24+lng.norm(A[:,r]))

        Diff = abs(np.dot(lng.inv(np.dot(A0.T,A0)),np.dot(A0.T,A)))

        if n_e > n:

            # the number of source is over-estimated
            # Loop on the sources

            IndTot = np.argsort(np.max(Diff,axis=1))[::-1]
            ind = np.linspace(0,n-1,n)

            m_select = np.ones((n_e,))

            for ns in range(0,n):
                indS = np.where(m_select > 0.5)[0]
                indix = np.where(Diff[IndTot[ns],indS] == max(Diff[IndTot[ns],indS]))[0]
                ind[IndTot[ns]] = indS[indix[0]]
                m_select[indS[indix[0]]] = 0

            Aq = A[:,ind.astype(int)]
            Sq = S[ind.astype(int),:]
            A0q = A0
            S0q = S0

        if n_e < n:

            # the number of source is under-estimated
            # Loop on the sources

            IndTot = np.argsort(np.max(Diff,axis=0))[::-1]
            ind = np.linspace(0,n_e-1,n_e)

            m_select = np.ones((n,))

            for ns in range(0,n_e):
                indS = np.where(m_select > 0.5)[0]
                indix = np.where(Diff[indS,IndTot[ns]] == max(Diff[indS,IndTot[ns]]))[0]
                ind[IndTot[ns]] = indS[indix[0]]
                m_select[indS[indix[0]]] = 0

            A0q = A0[:,ind.astype(int)]
            S0q = S0[ind.astype(int),:]
            Aq = A
            Sq = S

        IndE = ind.astype(int)

        #Aq = A[:,ind.astype(int)]
        #Sq = S[ind.astype(int),:]

    return A0q,S0q,Aq,Sq,IndE


################ CODE TO COMPUTE THE MIXING MATRIX CRITERION (AND SOLVES THE PERMUTATION INDETERMINACY)

def easy_CorrectPerm(cA0,cA):

    import numpy as np
    from scipy import special
    import scipy.linalg as lng
    import copy as cp
    from copy import deepcopy as dp
    import math

    A0 = cp.copy(cA0) # Reference matrix
    A = cp.copy(cA)

    for r in range(0,A0.shape[1]):
        A[:,r] = A[:,r]/(1e-24+lng.norm(A[:,r]))
        A0[:,r] = A0[:,r]/(1e-24+lng.norm(A0[:,r]))

    try:
        Diff = abs(np.dot(lng.inv(np.dot(A0.T,A0)),np.dot(A0.T,A)))
    except np.linalg.LinAlgError:
        Diff = abs(np.dot(np.linalg.pinv(A0),A))
        print('WARNING, PSEUDO INVERSE TO CORRECT PERMUTATIONS')

    ind = maximizeTrace(Diff)

    return A[:,ind],ind


###------------------------------------------------------------------------------###
################# CODE TO COMPUTE THE MIXING MATRIX CRITERION (AND SOLVES THE PERMUTATION INDETERMINACY)

def EvalCriterion_fast(A0,S0,A,S):

    # from copy import deepcopy as dp
    import numpy as np

    n = np.shape(A0)[1]
    n_e = np.shape(A)[1]

    incomp = 0
    if abs(n-n_e) > 0:
        incomp=1

    gA0,gS0,gA,gS,IndE = CorrectPerm_fast(A0,S0,A,S,incomp=incomp)
    n = np.shape(gA0)[1]
    n_e = np.shape(gA)[1]
    Diff = abs(np.dot(np.linalg.inv(np.dot(gA0.T,gA0)),np.dot(gA0.T,gA)))
    # Diff2 = dp(Diff)
    pmin = np.min(abs(Diff)) # pmin = np.min(abs(Diff2))

    z = np.min([n,n_e])

    Diff = Diff - np.diag(np.diag(Diff))
    Diff = Diff.reshape((z**2,))

    p = (np.sum(Diff))/(z*(z-1))
    pmax = np.max(abs(Diff))
    # pmin = np.min(abs(Diff2))
    pmed = np.median(abs(Diff))

    Q = abs(np.arccos(np.sum(gA0*gA,axis=0))*180./np.pi)

    crit = {"ca_mean":p,"ca_med":pmed,"ca_max":pmax,"SAE":Q}

    return crit

###------------------------------------------------------------------------------###
################# Outlier distance calculation

def matrix_outlierDistance(A_FM = 0,Ai = 0):
    import numpy as np
    dist = np.abs(np.diag(np.dot(A_FM.T,Ai)))
    return np.amax(np.ones(np.shape(dist)) - dist)

def matrix_CorrectPermutations(As = 0, Ss = 0, Aref = 0, Sref = 0,sizeBlock = 0):
    # from utils_dgmca import CorrectPerm_fast
    import copy as cp
    import numpy as np

    As_ = cp.deepcopy(As)
    # Ss_ = cp.deepcopy(Ss) # FAST_

    if (Ss.shape[1]%sizeBlock) == 0:
        lstBlocks = np.arange(As.shape[2]+1)*sizeBlock
    else:
        lstBlocks = np.arange(As.shape[2])*sizeBlock
        lstBlocks = np.append(lstBlocks,(As.shape[2]-1)*sizeBlock + As.shape[2]%sizeBlock)

    # Correct permutations
    for it1 in range(As.shape[2]):
        S0 = Sref[:,lstBlocks[it1]:lstBlocks[it1+1]]
        cA = As_[:,:,it1]
        cS = Ss[:,lstBlocks[it1]:lstBlocks[it1+1]] # cS = Ss_[:,lstBlocks[it1]:lstBlocks[it1+1]]

        A0q,S0q,Aq,Sq,IndE  = CorrectPerm_fast(Aref,S0,cA,cS,incomp=0)

        As_[:,:,it1] = Aq 
        # Ss_[:,lstBlocks[it1]:lstBlocks[it1+1]] = Sq

    return As_

###------------------------------------------------------------------------------###
################# Frechet Mean main function

# Function to apply the Frechet Mean over the columns of a group of matrices.
# The columns are not in order so they have to be rearanged before regarding a reference matrix
def matrix_FrechetMean(As = 0, Ss = 0, Aref = 0, Sref = 0 ,sizeBlock = 0, w = 0):
#--- Input variables dimensions and values
    # dim As = [n_obs,n_s,numBlock]
    # dim Ss = [n_s,sizeBlock*numBlock]
    # dim Aref = [n_obs,n_s]
    # dim Sref = [n_s,sizeBlock*numBlock]
    # numBlock = As.shape[2]

#--- Import useful modules
    from FrechetMean import FrechetMean
    # from utils_dgmca import CorrectPerm_fast
    import numpy as np

#--- Main code   
    A_FMean = np.zeros(Aref.shape)

    if (Ss.shape[1]%sizeBlock) == 0:
        lstBlocks = np.arange(As.shape[2]+1)*sizeBlock
    else:
        lstBlocks = np.arange(As.shape[2])*sizeBlock
        lstBlocks = np.append(lstBlocks,(As.shape[2]-1)*sizeBlock + As.shape[2]%sizeBlock)
   
    # Correct permutations
    for it1 in range(As.shape[2]):
        S0 = Sref[:,lstBlocks[it1]:lstBlocks[it1+1]]
        cA = As[:,:,it1]
        cS = Ss[:,lstBlocks[it1]:lstBlocks[it1+1]]

        A0q,S0q,Aq,Sq,IndE  = CorrectPerm_fast(Aref,S0,cA,cS,incomp=0)

        As[:,:,it1] = Aq 
        Ss[:,lstBlocks[it1]:lstBlocks[it1+1]] = Sq
        
    # Do the Frechet Mean over each group of columns
    for it2 in range(As.shape[1]):
        
        if np.shape(As)[2] != 1:
            A_FMean[:,it2] = FrechetMean(As[:,it2,:],w=w)

        else:
            A_FMean[:,it2] = np.reshape(As[:,it2,:],len(As[:,it2,:]))
        
    return A_FMean

##---------------------------------------------##
# Function to apply the Frechet Mean over the columns of a group of matrices.
# The columns are not in order so they have to be rearanged before regarding a reference matrix
# The arangement is done only using the A matrix and not the S matrix
def easy_matrix_FrechetMean(As = 0, Aref = 0, w = 0):
#--- Input variables dimensions and values
    # dim As = [n_obs,n_s,numBlock]
    # dim Aref = [n_obs,n_s]

#--- Import useful modules
    from FrechetMean import FrechetMean
    # from utils_dgmca import CorrectPerm_fast
    import numpy as np

    A_FMean = np.zeros(Aref.shape)

    # Correct permutations
    for it1 in range(As.shape[2]):
        As[:,:,it1],ind = easy_CorrectPerm(Aref,As[:,:,it1]) 

# Do the Frechet Mean over each group of columns
    for it2 in range(As.shape[1]):
        if np.shape(As)[2] != 1:
            A_FMean[:,it2] = FrechetMean(As[:,it2,:],w=w)
        else:
            A_FMean[:,it2] = np.reshape(As[:,it2,:],len(As[:,it2,:]))
        
    return A_FMean


##---------------------------------------------##
# Function to apply the Frechet Mean over the columns of a group of matrices.
# The columns are not in order so they have to be rearanged before regarding a reference matrix
# The arangement is done only using the A matrix and not the S matrix
def easy_matrix_FrechetMean2(As = 0, Aref = 0, w = 0):
#--- Input variables dimensions and values
    # dim As = [n_obs,n_s,numBlock]
    # dim Aref = [n_obs,n_s]
    # dim w = [n_s,numBlock]

#--- Import useful modules
    from FrechetMean import FrechetMean
    # from utils_dgmca import CorrectPerm_fast
    import numpy as np

    A_FMean = np.zeros(Aref.shape)
    
    # Correct permutations
    for it1 in range(As.shape[2]):
        As[:,:,it1],ind = easy_CorrectPerm(Aref,As[:,:,it1])
        w[:,it1] = w[ind,it1]

# Do the Frechet Mean over each group of columns
    for it2 in range(As.shape[1]):
        if np.shape(As)[2] != 1:

            if np.sum(abs(w[it2,:])>1e-8) == 1:
                A_FMean[:,it2] = np.reshape(As[:,it2,abs(w[it2,:])>1e-8],As.shape[0])

            elif np.sum(abs(w[it2,:])>1e-8) == 0:
                A_FMean[:,it2] = Aref[:,it2]

            else:
                A_FMean[:,it2] = FrechetMean(As[:,it2,:],w=w[it2,:])

        else:
            A_FMean[:,it2] = np.reshape(As[:,it2,:],As.shape[0])#,len(As[:,it2,:]))
        
    return A_FMean


###------------------------------------------------------------------------------###
################# Conjugate gradient for estimating A

def unmix_cgsolve(MixMat,X,tol=1e-6,maxiter=100,verbose=0):

    import numpy as np
    from copy import deepcopy as dp

    b = np.dot(MixMat.T,X)
    A = np.dot(MixMat.T,MixMat)
    S = 0.*dp(b)
    r = dp(b)
    d = dp(r)
    delta = np.sum(r*r)
    delta0 = np.sum(b*b)
    numiter = 0
    bestS = dp(S)
    bestres = np.sqrt(delta/delta0)

    while ((numiter < maxiter) & (delta > tol**2*delta0)):

        # Apply At A
        q = np.dot(A,d)
        alpha = delta/np.sum(d*q)

        S = S + alpha*d

        if np.mod(numiter+1,50) == 0:
            # r = b - Aux*x
            r = b - np.dot(A,S)
        else:
            r = r - alpha*q

        deltaold = dp(delta)
        delta = np.sum(r*r)
        beta = delta/deltaold
        d = r + beta*d
        numiter = numiter + 1

        if np.sqrt(delta/delta0) < bestres:
            bestS = S
            bestres = np.sqrt(delta/delta0)

    return bestS


###------------------------------------------------------------------------------###
################# Outlier distance calculation
# Function to correct the missing values from the estimations of A in A_block. 

def matrixCompletion(A_block,A_mean):

    import numpy as np
    tol = 1e-4

    A_est = np.zeros(A_mean.shape)
    amp_coef = np.zeros([A_block.shape[1],A_block.shape[2]]) # Number of columns x Number of matrices.

#-- Fix amplitude scaling indetermination
    for it_m in range(A_block.shape[2]):
        for it_c in range(A_block.shape[1]):
            
            amps = A_block[:,it_c,it_m]/(A_mean[:,it_c]*1.)
            amps = amps[abs(amps)>tol]
            if np.sum(abs(amps)) != 0:
                # amp_coef[it_c,it_m] = np.median(amps)
                amp_coef[it_c,it_m] = np.mean(amps)
            else:
                amp_coef[it_c,it_m] = 1.


    for it_c in range(A_block.shape[1]):
        for it_l in range(A_block.shape[0]):
            elements = np.divide(A_block[it_l,it_c,:],amp_coef[it_c,:]*1.) # Correct amplitudes
            elements = elements[abs(elements)>tol]
            if len(elements) != 0:
                # new_coef = np.median(elements)
                new_coef = np.mean(elements)
            else:
                new_coef = A_mean[it_l,it_c]
            A_est[it_l,it_c] = new_coef        
            
    for it_m in range(A_block.shape[2]):
        aux_mat = A_block[:,:,it_m]
        aux_mat[abs(aux_mat)<tol] = A_est[abs(aux_mat)<tol]
        # Normalizing columns in the estimation A_est
        aux_mat = aux_mat/np.maximum(1e-24,np.sqrt(np.sum(aux_mat*aux_mat,axis=0)))
        A_block[:,:,it_m] = aux_mat

    return A_block 











