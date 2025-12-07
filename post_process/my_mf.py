# https://github.com/atcemgil/notes/blob/master/Logistic%20Matrix%20Factorization.ipynb

import numpy as np
from sklearn.utils.extmath import randomized_svd

def sigmoid(t):
    return 1./(1+np.exp(-t))

def LogisticMF(Y, K, Mask=None, eta=0.001, nu=0.1, max_iter = 5000, log_step=-1, seed=0, init_std=0.005):
    M = Y.shape[0]
    N = Y.shape[1]

    W = np.random.default_rng(seed).normal(0, init_std,(M,K))
    H = np.random.default_rng(seed+1).normal(0, init_std,(K,N))

    YM = Y.copy()
    if Mask is not None:
        YM[Mask==False] = 0
    else:
        Mask = np.ones_like(YM)


    for epoch in range(max_iter):
        if epoch % 100 == 0:
            print(f'Epoch {epoch}')
        dLh = np.dot(W.T, YM-Mask*sigmoid(np.dot(W,H))) - nu*H
        H = H + eta * dLh
        dLw = np.dot(YM-Mask*sigmoid(np.dot(W,H)),H.T ) - nu*W
        W = W + eta * dLw

        if log_step > 0 and epoch % log_step == 0:
            LL = np.sum( (YM*np.log(sigmoid(np.dot(W,H))) +  (Mask-YM)*np.log(1 - sigmoid(np.dot(W,H)))) ) - nu*np.sum(H**2)/2. - nu*np.sum(W**2)/2. 
            print(epoch, LL)
        
    return W, H


def MF_PT(Y, r=5):
    U, S, VT = randomized_svd(Y, n_components=r)
    #print(U.shape, S.shape, VT.shape)
    W = np.matmul(U[:,:r],np.diag(np.sqrt(S[:r]))) # [M, K]
    H = np.matmul(np.diag(np.sqrt(S[:r])), VT[:r])
    return W, H
    
def MF_masked(Y, K, Mask, eta=0.001, nu=0.1, max_iter = 5000, log_step=-1, seed=0):
    M = Y.shape[0]
    N = Y.shape[1]

    W = np.random.default_rng(seed).normal(0,0.005,(M,K))
    H = np.random.default_rng(seed+1).normal(0,0.005,(K,N))

    YM = Y.copy()
    if Mask is not None:
        YM[Mask==False] = 0
    else:
        Mask = np.ones_like(YM)
    
    for epoch in range(max_iter):
        pred = np.dot(W, H) # [new_ocl, PT]
        pred[Mask == False] = 0

        dLw = np.dot(YM - pred, H.T) - nu * W
        W = W + eta * dLw

        pred = np.dot(W, H) # [new_ocl, PT]
        pred[Mask == False] = 0

        dLh = np.dot(W.T, YM - pred) - nu * H
        H = H + eta * dLh

        if log_step > 0 and epoch % log_step == 0:
            loss = (np.square(pred - YM)).sum() / Mask.sum() + nu * np.sum(W ** 2) / 2.  + nu * np.sum(H ** 2) / 2.
            print(epoch, loss)

    return W, H

def ContinualMF(W_prev, H_prev, new_Y, Mask, eta=0.001, nu=0.1, max_iter = 5000, log_step=-1, seed=0):
    # [ocl, K], [K, PT]
    W_new = np.stack([np.mean(W_prev, 0) for _ in range(new_Y.shape[0])]) # [new_ocl, K]
    
    new_YM = new_Y.copy()
    if Mask is not None:
        new_YM[Mask==False] = 0
    else:
        Mask = np.ones_like(new_YM)

    for epoch in range(max_iter):
        pred = np.dot(W_new, H_prev) # [new_ocl, PT]
        pred[Mask == False] = 0

        dLw = np.dot(new_YM - pred, H_prev.T) - nu * W_new
        W_new = W_new + eta * dLw

        if log_step > 0 and epoch % log_step == 0:
            loss = (np.square(pred - new_YM)).sum() / Mask.sum() + nu * np.sum(W_new ** 2) / 2. 
            print(epoch, loss)

    return W_new


def additive_pred_func(a, b):
    return a.reshape((-1,1)) + b.reshape((1,-1))

def LogisticMFAdditive(Y, Mask=None, eta=0.001, nu=0.1, max_iter = 5000, log_step=-1, seed=0):
    M = Y.shape[0]
    N = Y.shape[1]

    W = np.random.default_rng(seed).normal(0,1,M)
    H = np.random.default_rng(seed+1).normal(0,1,N)


    YM = Y.copy()
    if Mask is not None:
        YM[Mask==False] = 0
    else:
        Mask = np.ones_like(YM)


    for epoch in range(max_iter):
        dLh = np.sum((YM - Mask*sigmoid(additive_pred_func(W,H))), 0) - nu*H
        H = H + eta * dLh
        dLw = np.sum((YM - Mask*sigmoid(additive_pred_func(W,H))), 1) - nu*W
        W = W + eta * dLw

        if log_step > 0 and epoch % log_step == 0:
            LL = np.sum( (YM*np.log(sigmoid(additive_pred_func(W,H))) +  (Mask-YM)*np.log(1 - sigmoid(additive_pred_func(W,H)))) ) - nu*np.sum(H**2)/2. - nu*np.sum(W**2)/2. 
            print(epoch, LL)
        
    return W, H


def ContinualLogisticMFAdditive(W_prev, H_prev, new_Y, Mask, eta=0.001, nu=0.1, max_iter = 5000, log_step=-1, seed=0):
    # [ocl, K], [K, PT]
    W_new = np.stack([np.mean(W_prev, 0) for _ in range(new_Y.shape[0])]) # [new_ocl, K]
    
    new_YM = new_Y.copy()
    if Mask is not None:
        new_YM[Mask==False] = 0
    else:
        Mask = np.ones_like(new_YM)

    for epoch in range(max_iter):
        dLw = np.sum((new_YM - Mask*sigmoid(additive_pred_func(W_new,H_prev))), 1) - nu * W_new
        W_new = W_new + eta * dLw

        if log_step > 0 and epoch % log_step == 0:
            loss = np.sum( (new_YM*np.log(sigmoid(additive_pred_func(W_new,H_prev))) +  (Mask-new_YM)*np.log(1 - sigmoid(additive_pred_func(W_new,H_prev)))) ) \
                - nu*np.sum(H_prev**2)/2. - nu*np.sum(W_new**2)/2. 
            print(epoch, loss)

    return W_new
