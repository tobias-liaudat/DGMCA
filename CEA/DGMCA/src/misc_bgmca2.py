# Miscellaneous functions for DGMCA

import numpy as np
import scipy as sp
import copy as cp
from copy import deepcopy as dp
from starlet import *
from munkres import Munkres
from scipy import special
import scipy.signal as spsg
import scipy.linalg as lng
import math
from FrechetMean import FrechetMean
from utils2 import randperm
import warnings
warnings.filterwarnings("ignore", message="FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.")

# FUNCTIONS TO WORK ON THE TRANSFORMED DOMAIN

###------------------------------------------------------------------------------###
################# 2D array reformating into 1D array with sequential 2D square batches.
def reshapeDecImg(s,batch_size,J=0,normOpt=0,WNFactors=0):
#-- Function description
#   Takes the matrix s in 2D form and reshapes it into 1D form.
#   The reshape is done so that the batches of size batch_size which have to be as square as possible in 2D
#   are coded sequentially in the linear array.
#   Input:
#       s: 2D image
#       batch_size: number of pixels of the batch in the 2D image
#       J: Wavelet level decomposition. J=0 means no transformation.
#          The coarse resolution in the decomposition is not included
#       normOpt: 1: Activated. Normalize the different wavelet levels
#                0: Deactivated. No normalization. The levels are used just as they are calculated from the wavelet decomposition
#       WNFactors: Wavelet level factors. An array containing the scale value for each level (the variance of WGN wavelet decomposition level by level)
#       
#   The original 2D array can be reconstructed using the output variables.
#   Output:
#       line_id: The indices in the 1D array of the batchs
#       r_id: x and y indexes in order to reconstruct a 2D image from a 1D array
#          r_id[0,0,n] = x_first   ,   r_id[0,1,n] = x_last   ,   n : batch number
#          r_id[1,0,n] = y_first   ,   r_id[1,1,n] = y_last   ,   n : batch number
#       ccS: Coarse resolution coefficients in a 1D vector
#
#   Reconstruction (J=0): 
#   it = 0 ... n_batch
#   S_2D[r_id[0,0,n]:r_id[0,1,n],r_id[1,0,n]:r_id[1,1,n]] = 
#   np.reshape(S_1D[line_id[it]:line_id[it+1]],(r_id[0,1,n]-r_id[0,0,n],r_id[1,1,n]-r_id[1,0,n]))


#--- Import modules
    # import numpy as np
    # from starlet import *

#-- Main algorithm
#-- Defining block sizes for the reshaping
    if J < 0:
        print('J must be greater or equal to zero. Goodbye.')
        return
    
    nx = s.shape[0]
    ny = s.shape[1]
 
    xb_size = np.floor(np.sqrt(batch_size))
    yb_size = np.floor(np.sqrt(batch_size))

    xb_last = nx%xb_size
    yb_last = ny%yb_size
    nx_blocks = np.ceil(nx/(xb_size*1.)).astype(int)
    ny_blocks = np.ceil(ny/(yb_size*1.)).astype(int)

    n_blocks = nx_blocks*ny_blocks

    xb = np.arange(nx_blocks)*xb_size
    yb = np.arange(ny_blocks)*yb_size

    if xb_last > 0:
        xb = np.append(xb,(nx_blocks-1)*xb_size + xb_last)
    else:
        xb = np.append(xb,(nx_blocks)*xb_size)

    if yb_last > 0:
        yb = np.append(yb,(ny_blocks-1)*yb_size + yb_last)
    else:
        yb = np.append(yb,(ny_blocks)*yb_size)    

    xb = xb.astype(int)
    yb = yb.astype(int)
    
    
#-- Reshaping the data
    if J > 0:
        S = np.zeros((s.shape[0]*s.shape[1]*J,1))
        ccS = np.zeros((s.shape[0]*s.shape[1],1))
    else:
        S = np.zeros((s.shape[0]*s.shape[1],1))
        ccS = 0
        
    line_id = np.zeros(n_blocks+1)
    r_id = np.zeros((2,2,n_blocks))
    acc = 0
    accS = 0

    for it in range(n_blocks):
        ity = it%ny_blocks
        itx = np.floor(it/ny_blocks).astype(int)

        # The import can be done only for the batch, no need to import the whole matrix
        aux_s = s[xb[itx]:xb[itx+1],yb[ity]:yb[ity+1]]
        aux_size = aux_s.shape[0]*aux_s.shape[1]
        
        if J > 0:
            cc, ww = Starlet_Forward(x=aux_s,J=J)
            
            ccS[accS:accS+aux_size] = np.reshape(cc,(-1,1))
            accS += aux_size
            
            for itJ in range(J):
                
                if normOpt == 1:
                    norm_factor = WNFactors[0]/(WNFactors[itJ]*1.)
                else:
                    norm_factor = 1.
                    
                S[acc:acc+aux_size] = np.reshape(ww[:,:,itJ],(-1,1))*norm_factor
                acc += aux_size
                
        else:       
            S[acc:acc+aux_size] = np.reshape(aux_s,(-1,1))
            acc += aux_size
        
        line_id[it+1] = acc
        r_id[0,0,it] = xb[itx]
        r_id[0,1,it] = xb[itx+1]
        r_id[1,0,it] = yb[ity]
        r_id[1,1,it] = yb[ity+1]

    line_id = line_id.astype(int)
    r_id = r_id.astype(int)
    
    return S, line_id, r_id, ccS

###------------------------------------------------------------------------------###
################# Formatting a series of 2D images into a matrix where image will be a 1D array
def reshapeDecData(x,batch_size,J=0,normOpt=0,WNFactors=0):
#   Input:
#       x: 2D images stacked over the third axis
#       batch_size: number of pixels of the batch in the 2D image
#       J: Wavelet level decomposition. J=0 means no transformation.
#          The coarse resolution in the decomposition is not included
#       normOpt: 1: Activated. Normalize the different wavelet levels
#                0: Deactivated. No normalization. The levels are used just as they are calculated from the wavelet decomposition
#       WNFactors: Wavelet level factors. An array containing the scale value for each level (the variance of WGN wavelet decomposition level by level)
#       * len(WNFactors) >= J if normOpt=1, in order to be able to normalize
#
# 
#   Output:
#       X: 2D matrix where each line is a 2D image formated into an 1D array
#       line_id: The indices in the 1D array of the batchs
#       r_id: x and y indexes in order to reconstruct a 2D image from a 1D array
#          r_id[0,0,n] = x_first   ,   r_id[0,1,n] = x_last   ,   n : batch number
#          r_id[1,0,n] = y_first   ,   r_id[1,1,n] = y_last   ,   n : batch number
#       ccX: Coarse resolution coefficients matrix. Stacking 1D vector coefficients into a 2D matrix

#--- Import modules
    # import numpy as np
    # from starlet import *

#-- Main algorithm
    n = x.shape[2]
    if J > 0:
        X = np.zeros((n,x.shape[0]*x.shape[1]*J))
        ccX = np.zeros((n,x.shape[0]*x.shape[1]))
    else:    
        X = np.zeros((n,x.shape[0]*x.shape[1]))
        ccX = 0
    
    for it in range(n):
        x_it, line_id, r_id, ccX_it = reshapeDecImg(x[:,:,it],batch_size,J,normOpt,WNFactors)          
        X[it,:] = np.reshape(x_it,(x_it.shape[0]))
        
        if J > 0:
            ccX[it,:] = np.reshape(ccX_it,(ccX_it.shape[0]))

    return X, line_id, r_id, ccX


###------------------------------------------------------------------------------###
################# Recovering the 2D array from the 1D array following the reshaping of reshapeDecImg()
def recoverDecImg(X,r_id,line_id,J=0,ccX=0,deNormOpt=0,WNFactors=0):
#   Input:
#       X: 1D image (only one!)
#       line_id: The indices in the 1D array of the batchs
#       r_id: x and y indexes in order to reconstruct a 2D image from a 1D array
#          r_id[0,0,n] = x_first   ,   r_id[0,1,n] = x_last   ,   n : batch number
#          r_id[1,0,n] = y_first   ,   r_id[1,1,n] = y_last   ,   n : batch number
#       J: Wavelet level decomposition. J=0 means no transformation.
#          The coarse resolution in the decomposition is not included
#       ccX: Coarse resolution coefficients in a 1D vector
#       deNormOpt: 1: Activated. Normalize the different wavelet levels
#                  0: Deactivated. No normalization. The levels are used just as they are calculated from the wavelet decomposition
#       WNFactors: Wavelet level factors. An array containing the scale value for each level (the variance of WGN wavelet decomposition level by level)
#       * len(WNFactors) >= J if normOpt=1, in order to be able to normalize
# 
#   Output:
#       X: 2D matrix corresponding to the 2D image that was formated into an 1D array

#--- Import modules
    # import numpy as np
    # import copy as cp
    # from starlet import *
    
#-- Main algorithm
    n_batchs = r_id.shape[2]
    x_2D = np.zeros((r_id[0,1,n_batchs-1],r_id[1,1,n_batchs-1]))
    accS = 0
        
    
    for it in range(n_batchs):
        if J > 0:
            aux = X[line_id[it]:line_id[it+1]]
            
            dx = r_id[0,1,it]-r_id[0,0,it]
            dy = r_id[1,1,it]-r_id[1,0,it]
            block_size = dx*dy
            
            ww = np.zeros([dx,dy,J])
            
            cc = np.reshape(ccX[accS:accS+block_size],(dx,dy))
            accS += block_size
            
            acc = 0
            for itJ in range(J):
                if deNormOpt == 1:
                    deNorm_factor = WNFactors[itJ]/(WNFactors[0]*1.)
                else:
                    deNorm_factor = 1.
                
                ww[:,:,itJ] = np.reshape(aux[acc:acc+block_size],(dx,dy))*deNorm_factor
                acc += block_size
                
            x_2D[r_id[0,0,it]:r_id[0,1,it],r_id[1,0,it]:r_id[1,1,it]] = Starlet_Inverse(c=cc,w=ww)
                
        else:
            x_2D[r_id[0,0,it]:r_id[0,1,it],r_id[1,0,it]:r_id[1,1,it]] = \
                np.reshape(X[line_id[it]:line_id[it+1]],(r_id[0,1,it]-r_id[0,0,it],r_id[1,1,it]-r_id[1,0,it]))        
    
    return  x_2D


###------------------------------------------------------------------------------###
def recoverDecData(X,r_id,line_id,J=0,ccX=0,deNormOpt=0,WNFactors=0):
#   Input:
#       X: 1D images stacked into a 2D array. Each line is a 1D image
#       line_id: The indices in the 1D array of the batchs
#       r_id: x and y indexes in order to reconstruct a 2D image from a 1D array
#          r_id[0,0,n] = x_first   ,   r_id[0,1,n] = x_last   ,   n : batch number
#          r_id[1,0,n] = y_first   ,   r_id[1,1,n] = y_last   ,   n : batch number
#       J: Wavelet level decomposition. J=0 means no transformation.
#          The coarse resolution in the decomposition is not included
#       ccX: Coarse resolution coefficients in a 2D matrix, where each line is the CR of the corresponding
#           1D image
#       deNormOpt: 1: Activated. Normalize the different wavelet levels
#                  0: Deactivated. No normalization. The levels are used just as they are calculated from the wavelet decomposition
#       WNFactors: Wavelet level factors. An array containing the scale value for each level (the variance of WGN wavelet decomposition level by level)
#       * len(WNFactors) >= J if normOpt=1, in order to be able to normalize
#       WGN: White Gaussian Noise 
#
#   Output:
#       X: 2D matrices stacked over a third axis

#--- Import modules
    # import numpy as np

#-- Main algorithm
    n_imgs = X.shape[0]
    X_out = np.zeros((r_id[0,1,r_id.shape[2]-1],r_id[1,1,r_id.shape[2]-1],n_imgs))

    if np.isscalar(ccX): # if J == 0:
        for it in range(n_imgs):
            X_out[:,:,it] = recoverDecImg(X[it,:],r_id,line_id,J=J,ccX=0,deNormOpt=deNormOpt,WNFactors=WNFactors)
    else:
        for it in range(n_imgs):
            X_out[:,:,it] = recoverDecImg(X[it,:],r_id,line_id,J=J,ccX=ccX[it,:],deNormOpt=deNormOpt,WNFactors=WNFactors)


    return X_out

##
# FUNCTIONS TO PREPARE THE DATA TO WORK ON THE TRANSFORMED DOMAIN

def calcWaveletNoiseFactors(N=0, J=4):
# Calculation of the wavelet noise factors.

#--- Import modules
    # import numpy as np
    # from src.starlet import *
    # from src.utils2 import *
    # import copy as cp
    
    WNFactors = np.zeros([J])
    
    sqSize = np.ceil(np.sqrt(N.shape[0]*N.shape[1])).astype(int) 
    N_sq = cp.deepcopy(N)
    N_sq.resize(sqSize,sqSize) # Zero padding for the missing values
          
    cc, ww = Starlet_Forward(x=N_sq,J=J)

    for it in range(J):
        WNFactors[it] = np.std(np.reshape(ww[:,:,it],(-1)))
        
    
    return WNFactors


def input_prep(A_big,S_big,dec_factor,Jdec,normOpt,batch_size,n_obs,SNR_level):
    
    # Function parameters
    n_imgs = S_big.shape[0]
    img_dim = S_big.shape[1]/(dec_factor)
 
    #--------------------------------#
    # Data preparation for J = 0
    S0_J0 = np.zeros([img_dim,img_dim,n_imgs])

    for it in range(n_imgs):
    #     S0_J0[:,:,it] = S_big[it,::dec_factor,::dec_factor]
        new_im = spsg.decimate(S_big[it,:,:],dec_factor)
        new_im = spsg.decimate(new_im.T,dec_factor)
        S0_J0[:,:,it] = new_im.T

    S0_J0_mat, line_id_J0, r_id_J0, ccx = reshapeDecData(S0_J0,batch_size)

    id_obs = randperm(A_big.shape[0])[0:n_obs]
    A0_J0_mat = A_big[id_obs,:]

    X0_J0_mat = np.dot(A0_J0_mat,S0_J0_mat)

    # Adding observational noise
    N_J0 = np.random.randn(X0_J0_mat.shape[0],X0_J0_mat.shape[1])
    sigma_noise = np.power(10.,(-SNR_level/20.))*np.linalg.norm(X0_J0_mat,ord='fro')/np.linalg.norm(N_J0,ord='fro')
    N_J0 = sigma_noise*N_J0

    WNFactors = calcWaveletNoiseFactors(N=N_J0, J=5)

    X_J0_mat = X0_J0_mat + N_J0

    data_J0 = {'n_s': n_imgs, 'n_obs': n_obs, 'A0': A0_J0_mat, 'S0': S0_J0_mat, 'X0' : X0_J0_mat, 'X' : X_J0_mat,\
               'line_id' : line_id_J0, 'r_id' : r_id_J0, 'SNR':SNR_level, 'batch_size':batch_size,\
               'WNFactors' : WNFactors}
    
    #--------------------------------#
    # Data preparation for J = m

    X_Jm = np.zeros([img_dim,img_dim,X_J0_mat.shape[0]])

    for it in range(X_J0_mat.shape[0]):
        X_Jm[:,:,it] = recoverDecImg(X_J0_mat[it,:],r_id_J0,line_id_J0)
    
    
    #--------------------------------#
    # Data preparation for the original GMCA algorithm
    im_size = X_Jm.shape[0]*X_Jm.shape[1]

    X_GMCA = np.zeros([n_obs,im_size*Jdec])
    S0_GMCA = np.zeros([n_imgs,im_size*Jdec])

    for it in range(n_obs): 
        cc, ww = Starlet_Forward(x=X_Jm[:,:,it],J=Jdec)

        for itJ in range(Jdec):
            
            if normOpt == 1:
                norm_factor = WNFactors[0]/(WNFactors[itJ]*1.)
            else:
                norm_factor = 1.
            
            X_GMCA[it,im_size*itJ:im_size*(itJ+1)] = np.reshape(ww[:,:,itJ],(1,-1))*norm_factor


    for it in range(n_imgs):
        cc, ww = Starlet_Forward(x=S0_J0[:,:,it],J=Jdec)

        for itJ in range(Jdec):
            
            if normOpt == 1:
                norm_factor = WNFactors[0]/(WNFactors[itJ]*1.)
            else:
                norm_factor = 1.
            
            S0_GMCA[it,im_size*itJ:im_size*(itJ+1)] = np.reshape(ww[:,:,itJ],(1,-1))*norm_factor
    
    #--------------------------------#
    data_GMCA = {'n_s': n_imgs, 'n_obs': n_obs, 'A0': A0_J0_mat, 'S0': S0_J0_mat, 'X_GMCA' : X_GMCA, \
           'X' : X_Jm, 'line_id' : line_id_J0, 'r_id' : r_id_J0, 'SNR':SNR_level, 'batch_size':batch_size,\
           'WNFactors' : WNFactors, 'X0' : X0_J0_mat,'S0_GMCA' : S0_GMCA, 'normOpt' : normOpt}
    
    return data_GMCA
    

##
# FUNCTIONS TO ESTIMATE THE ALPHA (THRESHOLD) PARAMETER

###------------------------------------------------------------------------------###
 ################# Functions to estimate the Generalized Gaussian parameters
def GG_g_func(x,b,mu):
#--- Import modules
    # import scipy as sp
    # import numpy as np

#-- Main algorithm
    mu = mu *np.ones(x.shape)
    num = np.sum(((abs(x-mu))**b)*(np.log(abs(x-mu))))
    den = np.sum((abs(x-mu))**b) 
    
    return (1. + sp.special.polygamma(0,1/b)/b - num/den + np.log((b/len(x))*den)/b )


def GG_gprime_func(x,b,mu):
#--- Import modules
    # import scipy as sp
    # import numpy as np

#-- Main algorithm
    mu = mu *np.ones(x.shape)
    
    result = 1/(b**2) - sp.special.polygamma(0,1/b)/(b**2) - sp.special.polygamma(1,1/b)/(b**3)
    
    den = np.sum((abs(x-mu))**b)
    num1 = np.sum(((abs(x-mu))**b)*(np.log(abs(x-mu))))
    num2 = np.sum(((abs(x-mu))**b)*(np.log(abs(x-mu)))**2)
    
    return result - num2/den + (num1/den)**2 + num1/(b*den) - np.log((b/len(x))*den)/(b**2)


def GG_parameter_estimation(x,optHalf = 0):
#--- Import modules
    # import numpy as np
    # import scipy as sp

#-- Main algorithm    
    maxIts = 100 # Enough for convergence
    
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


def alpha_thresh_estimation(A_mean, X, alpha_exp,n):
#   Sources must not be thresholded for this estimation. 
#   Estimation sensible to the noise level of the observations.

#--- Import modules
    # import numpy as np

#-- Main algorithm    
    GG_mapping_param = np.array([ 4.5828 , -8.5807 ,  5.2610]) # Pre-calculed mapping model
    GG_alpha_est = np.zeros([n])

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
        GG_alpha_est = alpha_exp # We stay with the default value

#-- Mapping the rho estimate with the alpha_exp value       
    GG_alpha_est = GG_mapping_param[0]*(GG_alpha_est**2) + GG_mapping_param[1]*GG_alpha_est \
                    + GG_mapping_param[2]*np.ones(GG_alpha_est.shape) # Use of a quadratic model

#-- Thresholding values to avoid divergences (Model valid in that interval)
    if GG_alpha_est > 20.:
        GG_alpha_est = 20.
    elif GG_alpha_est < 0.01:
        GG_alpha_est = 0.01

    return GG_alpha_est



###------------------------------------------------------------------------------###
################# Calculation of pseudo-inverse with threshold in the singular values
def pInv(A):
#--- Import modules
    # import numpy as np

#-- Main algorithm
    min_cond = 1e-9
    Ra = np.dot(A.T,A)
    Ua,Sa,Va = np.linalg.svd(Ra)
    Sa[Sa < np.max(Sa)*min_cond] = np.max(Sa)*min_cond
    iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))

    return np.dot(iRa,A.T)



###------------------------------------------------------------------------------###
 ################# Calculate the column permutation that will give the maximum trace

def maximizeTrace(mat):
#--- Import modules
    # import numpy as np
    # import copy as cp
    # from munkres import Munkres
   
#-- Main algorithm     
    assert mat.shape[0] == mat.shape[1] # Matrix should be square
#--- Use of the Kuhn-Munkres algorithm (or Hungarian Algorithm)    
    m = Munkres()
    # Convert the cost minimization problem into a profit maximization one
    costMatrix = np.ones(np.shape(mat))*np.amax(mat) - mat       
    indexes = m.compute(cp.deepcopy(costMatrix))
    
    indexes= np.array(indexes)
    
    return (indexes[:,1]).astype(int)


###------------------------------------------------------------------------------###
################ Correct the permutation indeterminancy

def DGMCA_CorrectPerm(cA0,cA):

    # import numpy as np
    # from scipy import special
    # import scipy.linalg as lng
    # import copy as cp
    # from copy import deepcopy as dp
    # import math

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
# Function to apply the Frechet Mean over the columns of a group of matrices.
# The columns are not in order so they have to be rearanged before regarding a reference matrix
# The arangement is done by using only the A matrix and not the S matrix
def DGMCA_FrechetMean(As = 0, Aref = 0, w = 0):
#--- Input variables dimensions and values
    # dim As = [n_obs,n_s,numBlock]
    # dim Aref = [n_obs,n_s]
    # dim w = [n_s,numBlock]

#--- Import useful modules
    # from FrechetMean import FrechetMean
    # import numpy as np

#--- Main code
    A_FMean = np.zeros(Aref.shape)
    
#-- Correct permutations
    for it1 in range(As.shape[2]):
        As[:,:,it1],ind = DGMCA_CorrectPerm(Aref,As[:,:,it1])
        w[:,it1] = w[ind,it1]

#-- Do the Frechet Mean over each group of columns
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



### Function used in the performance evaluation ###

###------------------------------------------------------------------------------###
################# CODE TO COMPUTE THE MIXING MATRIX CRITERION (AND SOLVES THE PERMUTATION INDETERMINACY)

def CorrectPerm_eval(cA0,S0,cA,S,incomp=0):
#--- Import modules
    # import numpy as np
    # from scipy import special
    # import scipy.linalg as lng
    # import copy as cp
    # from copy import deepcopy as dp
    # import math

#-- Main algorithm    
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



###------------------------------------------------------------------------------###
################# CODE TO COMPUTE THE MIXING MATRIX CRITERION (AND SOLVES THE PERMUTATION INDETERMINACY)

def EvalCriterion_eval(A0,S0,A,S):

    # from copy import deepcopy as dp
    # import numpy as np

    n = np.shape(A0)[1]
    n_e = np.shape(A)[1]

    incomp = 0
    if abs(n-n_e) > 0:
        incomp=1

    gA0,gS0,gA,gS,IndE = CorrectPerm_eval(A0,S0,A,S,incomp=incomp)
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
    # import numpy as np
    dist = np.abs(np.diag(np.dot(A_FM.T,Ai)))
    return np.amax(np.ones(np.shape(dist)) - dist)

def matrix_CorrectPermutations(As = 0, Ss = 0, Aref = 0, Sref = 0,sizeBlock = 0):
    # import copy as cp
    # import numpy as np

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

        A0q,S0q,Aq,Sq,IndE  = CorrectPerm_eval(Aref,S0,cA,cS,incomp=0)

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
    # from FrechetMean import FrechetMean
    # import numpy as np

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

        A0q,S0q,Aq,Sq,IndE  = CorrectPerm_eval(Aref,S0,cA,cS,incomp=0)

        As[:,:,it1] = Aq 
        Ss[:,lstBlocks[it1]:lstBlocks[it1+1]] = Sq
        
    # Do the Frechet Mean over each group of columns
    for it2 in range(As.shape[1]):
        
        if np.shape(As)[2] != 1:
            A_FMean[:,it2] = FrechetMean(As[:,it2,:],w=w)

        else:
            A_FMean[:,it2] = np.reshape(As[:,it2,:],len(As[:,it2,:]))
        
    return A_FMean


###------------------------------------------------------------------------------###
################# Conjugate gradient for estimating A

def unmix_cgsolve(MixMat,X,tol=1e-6,maxiter=100,verbose=0):

    # import numpy as np
    # from copy import deepcopy as dp

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

    # import numpy as np
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

