import numpy as np
from datetime import timedelta # https://docs.python.org/3/library/datetime.html

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
    tmin, tmax = 0, 4*7+3
    Nmin = tmax
    Nmax = Nmin + 3*7
    N = np.random.randint(Nmin, Nmax+1)
    t0 = np.random.randint(tmin, tmax+1) # in days
    
    Im0 = np.random.uniform(1,100)
    lambda1 = np.random.uniform(10e-4,10e-3)
    
    return [pa, pIH, pIU, pHD, pHU, pUD, NI, NH, NU, R0, mu, N, t0, Im0, lambda1]

    