
import numpy as np



def Gp(x,x0,lp,approx,error): # Error in parabola 
    if error=="yes": err=1
    else: err=0
    if approx=="yes": appr=1
    else: appr=0
    Gp=(5e-4)*np.sin(np.pi*(x-x0)/lp)
    Gp=Gp*appr*err
    return Gp
    



def Gh(x,x0,lp,approx,error): # Error in hyperbola 
    if error=="yes": err=1
    else: err=0
    if approx=="yes": appr=1
    else: appr=0
    Gh=(5e-4)*np.sin(2*np.pi*(x-x0)/lp)
    Gh=Gh*appr*err
    return Gh



def d_Gp(x,x0,lp,approx,error): # Parabola error derivative
    if error=="yes": err=1
    else: err=0 
    if approx=="yes": appr=1
    else: appr=0
    d_Gp=(5e-4*np.pi/lp)*np.cos(np.pi*(x-x0)/lp)
    d_Gp=d_Gp*err
    return d_Gp



def d_Gh(x,x0,lp,approx,error): # Hyperbola error derivative
    if error=="yes": err=1
    else: err=0
    if approx=="yes": appr=1
    else: appr=0
    d_Gh=(5e-4*np.pi/lp)*np.cos(2*np.pi*(x-x0)/lp)*2
    d_Gh=d_Gh*err
    return d_Gh