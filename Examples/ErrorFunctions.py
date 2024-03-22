
import numpy as np


lp=840
x0=10000

def Gp(x): # Error in parabola 
    Gp=-(1e-3)*(x-x0-lp)*(x-x0)*0.2*4/lp**2
    return Gp
    



def Gh(x): # Error in hyperbola 
    Gh=-(1e-3)*(x-x0+lp)*(x-x0)*0.1*4/lp**2
    return Gh



def d_Gp(x): # Parabola error derivative
    d_Gp=-(1e-3)*(2*x-2*x0-lp)*0.2*4/lp**2
    return d_Gp



def d_Gh(x): # Hyperbola error derivative
    d_Gh=-(1e-3)*(2*x-2*x0+lp)*0.1*4/lp**2
    return d_Gh