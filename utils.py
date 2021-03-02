import numpy as np
from datetime import timedelta # https://docs.python.org/3/library/datetime.html

idx_to_keys = {
    0 : 'pa',
    1 : 'pIH',
    2 : 'pIU',
    3 : 'pHD',
    4 : 'pHU',
    5 : 'pUD',
    6 : 'NI',
    7 : 'NH',
    8 : 'NU',
    9 : 'R0',
    10 : 'mu',
    11 : 'N',
    12 : 't0',
    13 : 'Im0',
    14 : 'lambda1'
}

input_params = {
    'pa': [0.4,0.9],
    'pIH' : [0.05, 0.2],
    'pIU' : [0.01, 0.04],
    'pHD' : [0.1, 0.2],
    'pHU' : [0.1,0.2],
    'pUD' : [0.2, 0.4],
    'NI' : [8.0, 12.0],
    'NH' : [15.0,25.0],
    'NU' : [10., 20.],
    'R0' : [2.9, 3.4],
    'mu' : [0.01, 0.08],
    'N' : [4*7+3, 7*7+3],
    't0' : [0, 4*7+3],
    'Im0' : [1., 100.],
    'lambda1' : [10e-4, 10e-3]
}


def draw_input_params(seed = None):
    np.random.seed(seed)
    pa = np.random.uniform(low = 0.4, high = 0.9)
    pIH = np.random.uniform(0.05,0.2)
    pIU = np.random.uniform(0.01,0.04)
    pHD, pHU = np.random.uniform(0.1, 0.2, size=2)
    pUD = np.random.uniform(0.2,0.4)
    NI = np.random.uniform(8,12)
    NH = np.random.uniform(15,25)
    NU = np.random.uniform(10,20)
    R0 = np.random.uniform(2.9,3.4)
    mu = np.random.uniform(0.01,0.08)
    
    # we consider tmin to be the reference of time
    # each subsequent date is given relatively to tmin
    #tmax = timedelta(weeks = 4, days = 3)
    #Nmin = tmax
    #Nmax = Nmin + timedelta(weeks = 3)
    #N = Nmin+(1-np.random.uniform(0.0,1.0))*(Nmax-Nmin)
    #t0 = (1-np.random.uniform(0.0,1.0))*tmax
    #N = N.total_seconds()/86400
    #t0 = t0.total_seconds()/86400
    tmin, tmax = 0, 4*7+3 # 24/02/2020 = tmin = 0
    Nmin = tmax
    Nmax = Nmin + 3*7
    N = np.random.randint(Nmin, Nmax+1)
    t0 = np.random.randint(tmin, tmax+1) # in days
    
    Im0 = np.random.uniform(1,100)
    lambda1 = np.random.uniform(10e-4,10e-3)
    
    return [pa, pIH, pIU, pHD, pHU, pUD, NI, NH, NU, R0, mu, N, t0, Im0, lambda1]

def convert(arr):
    # [pa, pIH, pIU, pHD, pHU, pUD, NI, NH, NU, R0, mu, N, t0, Im0, lambda1]
    for k in range(arr.shape[0]):
        p = input_params[idx_to_keys[k]]
        arr[k] = p[0]+(p[1]-p[0])*arr[k]
        if(type(p[0]) == type(1) and type(p[1]) == type(1)):
            arr[k] = int(arr[k])
    return arr

def vectorize(delta):
    # goal : returning a delta for each j in [0,14]
    # by default, we'll return delta*[max - min]
    delta_vect = np.zeros((15))
    for k in range(15):
        p = input_params[idx_to_keys[k]]
        delta_vect[k] = delta*(p[1]-p[0])
        #if(type(p[0]) == type(1) and type(p[1]) == type(1)):
        #    delta_vect[k] = int(delta_vect[k])
    return delta_vect
# for now we don't choose to return int here.
