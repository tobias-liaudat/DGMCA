# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 07:59:05 2017

@author: jbobin
"""

import numpy as np
import scipy.linalg as lng
from copy import deepcopy as dp

################################################
# Defines exp and log maps for useful manifolds
################################################

### Sn

def RotMatrix(theta):

    """
     Rotation matrix
    """

    M = np.zeros((2,2))
    M[0,0] = np.cos(theta)
    M[1,1] = np.cos(theta)
    M[0,1] = np.sin(theta)
    M[1,0] = -np.sin(theta)

    return M

def Exp_Sn(xref,v,theta):

    """
     Exp-map of the n-sphere
    """

    m = len(xref)
    F = np.zeros((m,2))
    F[:,0] = xref
    F[:,1] = v
    F = np.dot(F,RotMatrix(-theta))

    return F[:,0]

def Log_Sn(xref,x):

    """
     Log-map of the n-sphere
    """

    m = len(x)

    G = np.zeros((m,))
    Gv = np.zeros((m,))

    a = np.sum(x*xref)/np.sqrt(np.sum(xref**2)*np.sum(x**2)+1e-24)

    if a > 1:
        a = 1
    if a < -1:
        a = -1

    G = np.arccos(a)  # Computing the angles

    v = x - a*xref
    Gv = v / (1e-24 + np.linalg.norm(v))   # Unit vector in the tangent subspace

    return G,Gv


#
#
#

def Grad(w,x,X):

    [m,T]=np.shape(X)
    g = np.zeros((m,))

    for t in range(T):
        gt,gv = Log_Sn(x,X[:,t])
        g = g - w[t]*gt*gv

    return g

def FrechetMean(X,u = None,w=None,t = None,itmax = 100,tol=1e-12):

    [m,T]=np.shape(X)
    g = np.zeros((m,))

    if u is None:
        u = np.random.rand(m,)
    if t is None:
        t = 1. # gradient path length
    if w is None:
        # w = np.ones((m,+))
        # w = w / np.sum(w)
        w = np.ones((T,)) # CHECK CORRECTION
        w = w / np.sum(w)


    for it in range(itmax):

        g = Grad(w,u,X)
        theta = np.linalg.norm(g)
        if theta != 0:
            gv = g/theta
        else:
            gv=g
            # print('WARNING: theta = 0')
        u_new = Exp_Sn(u,-gv,t*theta)
        diff = np.linalg.norm(u - u_new)
        u = np.copy(u_new)

        if diff < tol:
            break

    return u


###########################################
#   EVALUATION CRITERIA
###########################################


def CorrectPerm(cA0,S0,cA,S,optEchAS=0):

    import copy as cp
    from numpy import linalg as lng
    A0 = cp.copy(cA0)
    A = cp.copy(cA)

    nX = np.shape(A0)

    for r in range(0,nX[1]):
        S[r,:] = S[r,:]*(1e-24+lng.norm(A[:,r]))
        A[:,r] = A[:,r]/(1e-24+lng.norm(A[:,r]))
        S0[r,:] = S0[r,:]*(1e-24+lng.norm(A0[:,r]))
        A0[:,r] = A0[:,r]/(1e-24+lng.norm(A0[:,r]))

    try:
        Diff = abs(np.dot(lng.inv(np.dot(A0.T,A0)),np.dot(A0.T,A)))
    except np.linalg.LinAlgError:
        Diff = abs(np.dot(np.linalg.pinv(A0),A))
        print('WARNING, PSEUDO INVERSE TO CORRECT PERMUTATIONS')

    Sq = np.ones(np.shape(S))
    ind = np.linspace(0,nX[1]-1,nX[1])

    for ns in range(0,nX[1]):
        indix = np.where(Diff[ns,:] == max(Diff[ns,:]))[0]
        ind[ns] = indix[0]

    Aq = A[:,ind.astype(int)]
    Sq = S[ind.astype(int),:]

    for ns in range(0,nX[1]):
        p = np.sum(Sq[ns,:]*S0[ns,:])
        if p < 0:
            Sq[ns,:] = -Sq[ns,:]
            Aq[:,ns] = -Aq[:,ns]

    if optEchAS==1:
        return Aq,Sq,A0,S0
    else:
        return Aq,Sq


def EvalCompSep(A0,S0,A,S):

    gA,gS,gA0,gS0 = CorrectPerm(A0,S0,A,S,optEchAS=1)
    p,pmed = EvalCriterion(gA0,gS0,gA,gS)
    res = decomposition_criteria(gA0,gS.T, gS0.T, 0.*gS0.T)

    sad = abs(np.arcsin(1. - np.sum(gA0*gA,axis=0)))
    sad_med = np.median(sad)
    sad_max = np.max(sad)
    sad = np.mean(sad)

    return {'MMC':p,'MMC_med':pmed,'SAR':res[0]['SAR'],'SIR':res[0]['SIR'],'SDR':res[0]['SDR'],'SDR_med':res[0]['SDR median'],'SAD':sad,'SAD_med':sad_med,'SAD_max':sad_max}



def EvalCriterion(A0,S0,A,S,optMedian=1,Select=None):

    gA,gS,gA0,gS0 = CorrectPerm(A0,S0,A,S,optEchAS=1)

    if Select is not None:
        gA = gA[:,Select]
        gA0 = gA0[:,Select]

    try:
        Diff = abs(np.dot(np.linalg.inv(np.dot(gA0.T,gA0)),np.dot(gA0.T,gA)))
    except np.linalg.LinAlgError:
        Diff = abs(np.dot(np.linalg.pinv(gA0),gA))
        print('ATTENTION, PSEUDO-INVERSE POUR LE CRITERE SUR A')

    Diff = Diff - np.diag(np.diag(Diff))

    z = np.shape(gA)

    pmed = np.median(Diff[Diff > 1e-24])
    p = np.sum(Diff[Diff > 1e-24])/(np.size(Diff[Diff > 1e-24]))

    return p,pmed

def decomposition_criteria(A0,Se, Sr, noise):
    r"""
    Computes the SDR of each couple reference/estimate sources.

    Inputs
    ------
    Se: numpy array
        estimateof column signals.
    Sr: numpy array
        reference  of column signals.
    noise: numpy array
        noise matrix cotaminating the data.

    Outputs
    ------
    criteria: dict
        value of each of the criteria.
    decomposition: dict
        decomposition of the estimated sources under target, interference,
        noise and artifacts.
    """
    # Correct PERMUTATIONS
    gA,gS = CorrectPerm(A0,Sr.T,A0,Se.T)

    # compute projections
    Sr = Sr / np.maximum(0,np.linalg.norm(Sr,axis=0))
    Se = gS.T / np.maximum(0,np.linalg.norm(gS.T,axis=0))

    pS = Sr.dot(np.linalg.lstsq(Sr, Se)[0])
    SN = np.hstack([Sr, noise])
    #pSN = SN.dot(np.linalg.lstsq(SN, Se)[0])  # this crashes on MAC
    pSN = SN.dot(np.linalg.lstsq(SN, Se)[0])
    eps = np.spacing(1)
    # compute decompositions
    decomposition = {}
    # targets
    decomposition['target'] = np.sum(Se * Sr, 0) * Sr   # Sr is normalized
    # interferences
    decomposition['interferences'] = pS - decomposition['target']
    # noise
    decomposition['noise'] = pSN - pS
    # artifacts
    decomposition['artifacts'] = Se - pSN
    # compute criteria
    criteria = {}
    # SDR: source to distortion ratio
    num = decomposition['target']
    den = decomposition['interferences'] +\
        (decomposition['noise'] + decomposition['artifacts'])
    norm_num_2 = np.sum(num * num, 0)
    norm_den_2 = np.sum(den * den, 0)
    criteria['SDR'] = np.mean(10 * np.log10(
        np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    criteria['SDR median'] = np.median(10 * np.log10(
        np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    # SIR: source to interferences ratio
    num = decomposition['target']
    den = decomposition['interferences']
    norm_num_2 = sum(num * num, 0)
    norm_den_2 = sum(den * den, 0)
    criteria['SIR'] = np.mean(10 * np.log10(
        np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    # SNR: source to noise ratio
    if np.max(np.abs(noise)) > 0:  # only if there is noise
        num = decomposition['target'] + decomposition['interferences']
        den = decomposition['noise']
        norm_num_2 = sum(num * num, 0)
        norm_den_2 = sum(den * den, 0)
        criteria['SNR'] = np.mean(10 * np.log10(
            np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    else:
        criteria['SNR'] = np.inf
    # SAR: sources to artifacts ratio
    if (noise.shape[1] + Sr.shape[1] < Sr.shape[0]):
        # if noise + sources form a basis, there is no "artifacts"
        num = decomposition['target'] +\
            (decomposition['interferences'] + decomposition['noise'])
        den = decomposition['artifacts']
        norm_num_2 = sum(num * num, 0)
        norm_den_2 = sum(den * den, 0)
        criteria['SAR'] = np.mean(10 * np.log10(
            np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    else:
        criteria['SAR'] = np.inf
    return (criteria, decomposition)
