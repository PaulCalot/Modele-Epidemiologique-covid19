import numpy as np
from datetime import timedelta # https://docs.python.org/3/library/datetime.html


# -------------------------------- X - Input params ------------------------- #
nb_inputs = 15
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
key_to_idx = {v : k for k, v in idx_to_keys.items()}
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
    delta_vect = np.zeros((15,15))
    for k in range(15):
        p = input_params[idx_to_keys[k]]
        delta_vect[k,k] = delta*(p[1]-p[0])
        #if(type(p[0]) == type(1) and type(p[1]) == type(1)):
        #    delta_vect[k] = int(delta_vect[k])
    return delta_vect
# for now we don't choose to return int here.

# sampling methods :
def Monte_Carlo_sampling(R = 100):
    list_samples = [draw_input_params() for k in range(R)]
    return np.array(list_samples)

def lhs(R = 100):
    from pyDOE2 import lhs # https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube
    lhs = lhs(15 , samples = R) # 15 dimension of the input
    # everything is in [0,1]
    
    # now putting it in the right format
    
    for k in range(R):
        lhs[k] = convert(lhs[k])
        
    return lhs
    
# -------------------------------------- Y - Output -------------------------- #
plotting_names = {
    'S' : 'Susceptible (S)',
    'Im' : 'Infec. not Detec. (I-)',
    'Ip' : 'Infec. Detec. (I+)',
    'Rm' : 'Recov. not Detec. (R-)',
    'RI' : 'Recov. Detec. (R+)',
    'H' : 'Hospital (H)',
    'U' : 'Intensive care (U)',
    'RH' : 'Recov. Hosp. (R+H)',
    'D' : 'Dead (D)',
    'DR' : 'Detected (DR)'
}
box_colors = {
    'S' : 'yellow',
    'Im' : 'purple',
    'Ip' : 'orange',
    'Rm' : 'olive',
    'RI' : 'olivedrab',
    'H' : 'lightblue',
    'U' : 'pink',
    'RH' : 'darkolivegreen',
    'D' : 'red',
    'DR' : 'blue'
}

ykeys = {
        'S' : 0,
        'Im' : 1,
        'Ip' : 2,
        'Rm' : 3,
        'RI' : 4,
        'H' : 5,
        'U' : 6,
        'RH' : 7,
        'D' : 8,
        'DR' : 9
    }

def extract_int_value(solution, step_in_day):
    # start:end:step
    sol, period = (solution[::int(1/step_in_day),:], 1) if step_in_day < 1 else (solution, step_in_day) # extraction of integer day value
    return sol, period # period = 1, if we have a value for each day, if one value every two days, equals 2 etc.

def get_max_load_intensive_care(solution, period):
    # order : S, Im, Ip, Rm, RI, H, U, RH, D, DR
    # we suppose that the first element is the 0th day.
    max_it = np.argmax(solution[:,ykeys['U']])
    max_U = solution[max_it,ykeys['U']]
    max_time = max_it*period
    return np.array([int(max_time), int(max_U)])
    