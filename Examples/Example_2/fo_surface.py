
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import interpolate


def fo_curve(x0,y0,l,theta_tar0):
  f=np.sqrt(x0**2+y0**2)
  alpha0=0.25*np.arctan(y0/x0)
  dx1=0.01
  x10=x0+dx1
  y10=y0+dx1*np.tan(alpha0)
  theta_tar=np.deg2rad(theta_tar0)
  x2_a0=((y10-y0)+x0*np.tan(3*alpha0)-x10*np.tan(2*alpha0-theta_tar))/(np.tan(3*alpha0)-np.tan(2*alpha0-theta_tar))
  y2_a0=y0+(x2_a0-x0)*np.tan(3*alpha0)
  beta_a0=(np.arctan((y2_a0+f*theta_tar)/x2_a0)+2*alpha0-theta_tar)/2

  #################################################
  dx2=-0.0001
  x2_a=x2_a0 ; y2_a=y2_a0
  alpha=alpha0; y1=y10; x1=x10; beta_a=beta_a0
  N=abs(int(l/dx2))
  X2=np.array([])
  Y2=np.array([])
  X1=np.array([]);Y1=np.array([])
  Beta=np.array([])
  i = 0
  theta_temp=0
  while theta_tar>=np.abs(theta_temp):
    X2=np.append(X2,x2_a)
    Y2=np.append(Y2,y2_a)
    X1=np.append(X1,x1)
    Y1=np.append(Y1,y1)
    Beta=np.append(Beta,beta_a)
    x2_a=x2_a+dx2
    y2_a=y2_a+np.tan(beta_a)*dx2
    theta_temp=-np.arctan((y2_a-y10)/(x2_a-x10))+2*alpha0
    beta_a=(np.arctan((y2_a+f*theta_temp)/x2_a)+2*alpha0-theta_temp)/2
    i+=1
  

  x2_b=X2[-1] ; y2_b=Y2[-1]; beta_b=Beta[-1]
  alpha=alpha0; y1=y10; x1=x10
  X1=np.array([]);Y1=np.array([]); Alpha=np.array([])
  i=1
  while x1<x0+l+10:
    X2=np.append(X2,x2_b)
    Y2=np.append(Y2,y2_b)
    Beta=np.append(Beta,beta_b)
    X1=np.append(X1,x1)
    Y1=np.append(Y1,y1)
    Alpha=np.append(Alpha,alpha)
    x2_a=X2[i]
    y2_a=Y2[i]
    beta_a=Beta[i]
    n=np.tan(alpha)
    alpha=(2*beta_a-np.arctan((y2_a+f*theta_tar)/x2_a)+theta_tar)/2
    m=np.tan(2*alpha-theta_tar)
    x1=(y1-y2_a+m*x2_a-n*x1)/(m-n)
    y1=m*(x1-x2_a)+y2_a
    x2_b=X2[-1]
    y2_b=Y2[-1]
    beta_b=Beta[-1]
    n1=np.tan(beta_b)
    m1=np.tan(2*alpha+theta_tar)
    x2_b=(y1-y2_b-m1*x1+n1*x2_b)/(n1-m1)
    y2_b=y1+m1*(x2_b-x1)
    beta_b=(np.arctan((y2_b-f*theta_tar)/x2_b)+2*alpha+theta_tar)/2
    i+=1

  y1_fun= interpolate.interp1d(X1,Y1)
  y2_fun= interpolate.interp1d(X2,Y2)

  d_y1_fun=  interpolate.interp1d(X1, np.tan(Alpha))
  d_y2_fun=  interpolate.interp1d(X2, np.tan(Beta))
  
  return y1_fun, d_y1_fun, y2_fun, d_y2_fun


x0=2000
r0=200
lp=30
theta_tar=0.25

yp_fun,d_yp_fun,yh_fun,d_yh_fun=fo_curve(x0,r0,lp,theta_tar)
    

def fo_Gp(x): # Error in parabola 
    Gp=np.zeros_like(x)
    index=np.where((x >yp_fun.x.min()) & (x <yp_fun.x.max()))
    Gp[index]=yp_fun(x[index])
    return Gp
    

def fo_Gh(x): # Error in hyperbola 
    Gh=np.zeros_like(x)
    index=np.where((x >yh_fun.x.min()) & (x <yh_fun.x.max()))
    Gh[index]=yh_fun(x[index])
    return Gh


def fo_d_Gp(x): # Parabola error derivative
    d_Gp=np.zeros_like(x)
    index=np.where((x >d_yp_fun.x.min()) & (x <d_yp_fun.x.max()))
    d_Gp[index]=d_yp_fun(x[index])
    return d_Gp


def fo_d_Gh(x): # Hyperbola error derivative
    d_Gh=np.zeros_like(x)
    index=np.where((x >d_yh_fun.x.min()) & (x <d_yh_fun.x.max()))
    d_Gh[index]=d_yh_fun(x[index])
    return d_Gh