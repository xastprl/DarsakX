
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import interpolate

def ws_curve(x0,r0,lp):
  lp=3*lp
  xp=np.linspace(x0+1e-2,x0+lp,10000)
  xh=np.linspace(x0-lp,x0-1e-2,10000)
  xi=1
  #######- WO Parameters
  alpha=0.25*np.arctan(r0/x0) # grazing angle in radian
  theta_p=2*xi*alpha/(1+xi)
  theta_h=2*(1+2*xi)*alpha/(1+xi)
  p=x0*np.tan(4*alpha)*np.tan(theta_p) # p/2 is the focal length of parabola mm
  d=x0*np.tan(4*alpha)*np.tan(4*alpha-theta_h)
  e=np.cos(4*alpha)*(1+np.tan(4*alpha)*np.tan(theta_h)) # eccentricity of hyperbola
  rp0=np.sqrt(p**2+2*p*(xp)+((4*e*e*p*d)/(e*e-1)))
  rh0=np.sqrt((e*(xh+d))**2-xh**2)
  #plt.plot(xp,rp0)
  #plt.plot(xh,rh0)

  ############### WS
  B0=np.arctan(r0/x0)
  f=x0/np.cos(B0)
  k=np.tan(B0/2)**2
  B=np.ones_like(xp)
  for i in range(len(xp)):
    def fun_p(B):
      fp=xp[i]+f*np.sin(B0/2)**2-(f*np.sin(B))**2/(4*f*np.sin(B0/2)**2)-f*(np.cos(B/2)**4)*(np.tan(B/2)**2/k-1)**(1-k)
      return fp
    B[i]=fsolve(fun_p, B[i])
  rp=f*np.sin(B)

  
  d=f/((1-np.cos(B))/(1-np.cos(B0))+((1+np.cos(B))/2)*(np.tan(B/2)**2/k-1)**(1+k))
  xh=d*np.cos(B)
  rh=d*np.sin(B)
  
  temp=(rp-d*np.sin(B))/(xp-d*np.cos(B))
  Alpha=np.arctan(temp)/2
  Gamma=B/2+Alpha
  
  X1=xp; Y1=rp ; X2=xh; Y2=rh
  
  d_y1_fun=  interpolate.interp1d(X1, np.tan(Alpha) )
  d_y2_fun=  interpolate.interp1d(X2,np.tan(Gamma) )
  y1_fun= interpolate.interp1d(X1,Y1)
  y2_fun= interpolate.interp1d(X2,Y2)

  return y1_fun, d_y1_fun, y2_fun, d_y2_fun


x0=2000
r0=200
lp=30

yp_fun,d_yp_fun,yh_fun,d_yh_fun=ws_curve(x0,r0,lp)
    

def ws_Gp(x): # Error in parabola 
    Gp=np.zeros_like(x)
    index=np.where((x >yp_fun.x.min()) & (x <yp_fun.x.max()))
    Gp[index]=yp_fun(x[index])
    return Gp
    

def ws_Gh(x): # Error in hyperbola 
    Gh=np.zeros_like(x)
    index=np.where((x >yh_fun.x.min()) & (x <yh_fun.x.max()))
    Gh[index]=yh_fun(x[index])
    return Gh


def ws_d_Gp(x): # Parabola error derivative
    d_Gp=np.zeros_like(x)
    index=np.where((x >d_yp_fun.x.min()) & (x <d_yp_fun.x.max()))
    d_Gp[index]=d_yp_fun(x[index])
    return d_Gp


def ws_d_Gh(x): # Hyperbola error derivative
    d_Gh=np.zeros_like(x)
    index=np.where((x >d_yh_fun.x.min()) & (x <d_yh_fun.x.max()))
    d_Gh[index]=d_yh_fun(x[index])
    return d_Gh