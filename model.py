import numpy as np
from scipy.integrate import odeint

#from datetime import timedelta # https://docs.python.org/3/library/datetime.html
from pprint import pprint
from collections import OrderedDict
from datetime import timedelta, datetime

from utils import delta_time, key_to_idx, ykeys, init_date
from utils import extract_int_value, fill_missing_values

# SEIR
class SEIR:
    
    S0 = 67e6 # size of french population
    
    keys = key_to_idx
    
    rkeys = {
        'gIR' : 0,
        'gIH' : 1,
        'gIU' : 2,
        'gHD' : 3,
        'gHU' : 4,
        'gHR' : 5,
        'gUD' : 6,
        'gUR' : 7
    }
    
    ykeys = ykeys
    
    def __init__(self, x):
        # x contains the fifteen input parameter
        # order : pa, pIH, pIU, pHD, pHU, pUD, NI, NH, NU, R0, mu, N, t0, Im0, lambda1  # 15
        self.x = x
        # order : S, Im, Ip, Rm, RI, H, U, RH, D, DR # 10
        self.y = np.array([self.S0, x[self.keys['Im0']], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.t = x[self.keys['t0']]
        
        self.rates = self.init_rates()
        #print(self.rates)
        self.tau0 = x[self.keys['R0']]*(x[self.keys['lambda1']]+\
                        self.rates[self.rkeys['gIR']]+\
                        self.rates[self.rkeys['gIH']]+\
                        self.rates[self.rkeys['gIU']])/self.S0
    
    # step
    def step(self, dt):
        self.y += dt*self.derivative(self.y, self.t) # Euler explicit
        self.t += dt

    # derivative
    def derivative(self, y, t):
        der = np.zeros((11))
        tau = self.get_tau(t)
    
        S, Im, Ip, Rm, RI, H, U, RH, D, DR = list(y)
        gIR, gIH, gIU, gHD, gHU, gHR, gUD, gUR = self.rates
        l1 = self.x[self.keys['lambda1']]
        
        return np.array((
            -tau*S*Im, # dS
            tau*S*Im - l1*Im - (gIR+gIH+gIU)*Im, # dIm
            l1*Im - (gIR+gIH+gIU)*Ip, # dIp
            gIR*Im, # dRm 
            gIR*Ip, # dRI
            gIH*(Im+Ip)-(gHR+gHD+gHU)*H, # dH
            gIU*(Im+Ip)+gHU*H-(gUR+gUD)*U, # dU
            gHR*H+gUR*U, # dRH
            gUD*U+gHD*H, # dD
            (l1+gIH+gIU)*Im-DR # dDR
        ))
        
    def init_rates(self):
        # order : gIR, gIH, gIU, gHD, gHU, gHR, gUD, gUR
        pa, pIH, pIU, pHD, pHU, pUD, NI, NH, NU = self.x[:9]
        
        gIR = (pa+(1-pa)*(1-pIH-pIU))/NI
        gIH = (1-pa)*pIH/NI
        gIU = (1-pa)*pIU/NI
        gHD = pHD/NH
        gHU = pHU/NH
        gHR = (1-pHD-pHU)/NH
        gUD = pUD/NU
        gUR = (1-pUD)/NU
        
        return [gIR, gIH, gIU, gHD, gHU, gHR, gUD, gUR]
        
    # ---------------- getter and setter --------------- #
    def get_state(self):
        return self.y
    
    def get_fcn(self):
        return lambda y, t : self.derivative(y,t)
    
    def get_tau(self, t):
        dtime = (t-self.x[self.keys['N']])
        return self.get_tau0(t)*np.exp(-self.x[self.keys['mu']]*\
                    max(dtime,0))
    
    def get_tau0(self,t):
        x=self.x
        if(t>self.x[self.keys['N']]):
            tau0 = x[self.keys['R0']]*(self.rates[self.rkeys['gIR']]+\
                        self.rates[self.rkeys['gIH']]+\
                        self.rates[self.rkeys['gIU']])/self.S0
        else:
            tau0 = x[self.keys['R0']]*(x[self.keys['lambda1']]+\
                        self.rates[self.rkeys['gIR']]+\
                        self.rates[self.rkeys['gIH']]+\
                        self.rates[self.rkeys['gIU']])/self.S0
        return tau0
    # --------------- print functions ------------------ #
    def __str__(self):
        dico_params = self.params_to_dict(self.keys, self.x)
        dico_rates = self.params_to_dict(self.rkeys, self.rates)
        dico_system_state = self.params_to_dict(self.ykeys, self.y)
        return 'Model of type SEIR with : \n\t- input parameters : {} \n\t- Rates : {} \nt\t- State : {} \n\t - Current time {}'.format(dico_params, dico_rates, dico_system_state, self.t)
    
    # prettyprint
    def prettyprint(self):
        dico_params = self.params_to_dict(self.keys, self.x)
        dico_rates = self.params_to_dict(self.rkeys, self.rates)
        dico_system_state = self.params_to_dict(self.ykeys, self.y)
        print('{:=^40}'.format(' SEIR MODEL '))
        print('{:-^40}'.format(' input parameters '))
        pprint(dico_params)
        print('{:-^40}'.format(' Rates '))
        pprint(dico_rates)        
        print('{:-^40}'.format(' State '))
        pprint(dico_system_state)
        print('{:-^40}'.format(' Current time '))
        print(self.t)
        print('{:=^40}'.format(''))

    def params_to_dict(self, positions_dict, vector):
        dico = OrderedDict()
        for key, value  in positions_dict.items():
            dico[key] = vector[value]
        return dico


    
# functions used to quikcly get integer results from input_params (uses precedes)

def f(input_params, tend = datetime(year = 2020, month = 5, day = 11), verbose = False):    
    model = SEIR(input_params)
    
    if(verbose):
        model.prettyprint()
        
    fcn = model.get_fcn()
    y_ini = model.get_state()

    # in number of days
    tini = int(input_params[key_to_idx['t0']])
    init_date_simu = init_date+tini*timedelta(days = 1)
    
    t_simu = np.arange(tini, tini+(tend-init_date_simu).days+1)
    
    rtol, atol = 1e-3, 1e-6 # default values
    solution = odeint(func = fcn, t = t_simu, y0 = y_ini) 
    
    # we will make so that simulations always start from the same initialization time
    sol = fill_missing_values(solution, number_missing = tini, filling_value = 'first')
    return sol
