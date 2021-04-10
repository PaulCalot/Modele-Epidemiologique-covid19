from utils import idx_to_keys, input_params
from utils import lhs, Monte_Carlo_sampling
from utils import nb_inputs
from model import f

import numpy as np
# --------------------------- Q3) Morris ------------------------- # 

def compute_finite_diff(samples, delta, fn, h):
    from utils import vectorize
    delta_vect = vectorize(delta) # delta*np.identity(nb_inputs) #
    arr = np.zeros((len(samples), nb_inputs, h))
    df_x = [fn(f(sample)) for sample in samples]
    for r, sample in enumerate(samples): # loop of size R
        for idx, d in enumerate(delta_vect): # d = [0,...,0, delta_idx, 0, ..., 0], d_idx = delta_idx
            p = input_params[idx_to_keys[idx]]
            xdx = sample+d 
            while(xdx[idx]<p[0] or xdx[idx]>p[1]):
                d = 0.5*d
                xdx = sample+d
            df = (fn(f(xdx))-df_x[idx])/d[idx]
            arr[r, idx] = df
    return arr

def Morris(fn, h, R = 100, delta = 1e-3, method = 'lhs'):
    if(method == 'lhs'):
        samples = lhs(R=R)
    elif(method == 'MC'):
        samples = Monte_Carlo_sampling(R=R)
    else:
        print("Method {} not recognized. \nChoices : 'lhs', 'MC'".format(method))
        return
    
    fd = compute_finite_diff(samples, delta, fn, h)
    fd = np.transpose(fd) #  np.moveaxis(fd, [2, 0], [0, 3])
    
    mu_arr = np.mean(np.abs(fd), axis = -1) 
    mean = np.mean(fd, axis = -1)
    sigma_arr = np.std(fd, axis = -1)

    return mu_arr, sigma_arr  

# --------------------------- Q4) Sobol ------------------------- # 

def Sobol(fn, R = 100, method = 'lhs'):
    if(method == 'lhs'):
        A,B = lhs(R=R), lhs(R=R) # not how we are suppose to do it I think.
    elif(method == 'MC'):
        A,B = Monte_Carlo_sampling(R=R),Monte_Carlo_sampling(R=R)
    else:
        print("Method {} not recognized. \nChoices : 'lhs', 'MC'".format(method))
        return
    
    fA = np.array([fn(f(X)) for X in A])
    fB = np.array([fn(f(X)) for X in B])

    s = list(fA.shape)
    s[0] = nb_inputs
    S = np.zeros(s) 
    St = np.zeros(s)

    mu_tot = np.mean(fA, axis = 0) 
    sigma_tot = np.var(fA, axis = 0)
    
    for i in range(nb_inputs):
        Ci = np.copy(A)
        Ci[:,i] = B[:,i]
        fCi = np.array([fn(f(X)) for X in Ci])
        
        Vhat_i = np.mean(fB*(fCi-fA), axis = 0) 
        Vhat_mi = np.mean(fA*(fA-fCi), axis = 0)
        
        S[i] = Vhat_i/sigma_tot
        St[i] = Vhat_mi/sigma_tot
        
    return S, St