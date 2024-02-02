import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def wo_ray_trace(r0,x0,xi,theta,lp,lh,approx,error,dl,xd,Gp,Gh,dGp,dGh):
    
    # r0 mirror radius in mm
    # x0 focal lenght in mm
    # ratio of incident angle 
    # thetha off-axis angle deg
    # lp lenght of parabola in mm
    

    
    #######- WO Parameters
    alpha=0.25*np.arctan(r0/x0) # grazing angle in radian
    theta_p=2*xi*alpha/(1+xi)
    theta_h=2*(1+2*xi)*alpha/(1+xi)
    p=x0*np.tan(4*alpha)*np.tan(theta_p) # p/2 is the focal length of parabola mm
    d=x0*np.tan(4*alpha)*np.tan(4*alpha-theta_h)
    e=np.cos(4*alpha)*(1+np.tan(4*alpha)*np.tan(theta_h)) # eccentricity of hyperbola
    theta=theta*np.pi/180
    #######

    ######-Off-Axis Source Grid
    lmin_top=r0*np.cos(theta)/np.sqrt(2)+lp*np.sin(theta)
    lmin_bottom=-r0*np.cos(theta)/np.sqrt(2)+lp*np.sin(theta)
    lmax=r0+lp*np.tan(theta_p)
    zi_B=np.concatenate((-np.arange(dl,r0*np.cos(theta)/np.sqrt(2),dl),np.arange(0,r0*np.cos(theta)/np.sqrt(2),dl)))
    yi_A=np.arange(-lmax,lmax,dl)
    yi_B=np.concatenate((np.arange(-lmax,lmin_bottom,dl),-np.arange(-np.max(yi_A),-lmin_top,dl)))
    zi_A=np.concatenate((-np.arange(np.max(zi_B),lmax,dl),np.arange(np.max(zi_B),lmax,dl)))
    zzi_A, yyi_A = np.meshgrid(zi_A, yi_A)
    zzi_B, yyi_B = np.meshgrid(zi_B, yi_B)
    zzi=np.concatenate((zzi_A.ravel(),zzi_B.ravel()))
    yyi=np.concatenate((yyi_A.ravel(),yyi_B.ravel()))
    ###################
    

    ##############-Light filter after 1st reflection at parabola 
    if error=='no'or (error=='yes' and approx=='yes'): 
      Rix,Riy,Riz=(x0+lp-yyi*np.sin(theta)).ravel(), (yyi*np.cos(theta)).ravel(), zzi.ravel()
      n_ip=np.array([-np.cos(theta), -np.sin(theta),0])
      Ai=(n_ip[1]**2)/(n_ip[0]**2)
      Bi=2*(Riy-Rix*(n_ip[1])/(n_ip[0]))*(n_ip[1])/(n_ip[0])-2*p
      Ci=Riz**2+(Riy-Rix*(n_ip[1])/(n_ip[0]))**2-p**2-((4*e*e*p*d)/(e*e-1))
      if theta==0:
       xp=-Ci/Bi
      else:
       xp=np.zeros_like(Bi)
       index_xp=np.where((Bi**2-4*Ai*Ci)>0)
       xp[index_xp]=(-Bi[index_xp]-np.sqrt(Bi[index_xp]**2-4*Ai*Ci[index_xp]))/2/Ai
      rp=np.sqrt(p**2+2*p*(xp)+((4*e*e*p*d)/(e*e-1)))
      sin_phi_p=Riz/rp
      cos_phi_p=(Riy+(xp-Rix)*(n_ip[1])/(n_ip[0]))/rp
      phi_p=np.arctan2(sin_phi_p,cos_phi_p)
      indexi=np.where(((xp>=x0))  & (xp<=x0+lp))
      rp=rp[indexi]
      phi_p=phi_p[indexi]
      xp=xp[indexi]
      rp=np.sqrt(p**2+2*p*(xp)+((4*e*e*p*d)/(e*e-1)))
    elif error=='yes' and approx=='no':
      Rix,Riy,Riz=(x0+lp-yyi*np.sin(theta)).ravel(), (yyi*np.cos(theta)).ravel(), zzi.ravel()
      n_ip=np.array([-np.cos(theta), -np.sin(theta),0])
      Ai0=(n_ip[1]**2)/(n_ip[0]**2)
      Bi0=2*(Riy-Rix*(n_ip[1])/(n_ip[0]))*(n_ip[1])/(n_ip[0])-2*p
      Ci0=Riz**2+(Riy-Rix*(n_ip[1])/(n_ip[0]))**2-p**2-((4*e*e*p*d)/(e*e-1))
      if theta==0:
       xp=-Ci0/Bi0
      else:
       xp=np.ones_like(Bi0)*x0
       index_xp=np.where((Bi0**2-4*Ai0*Ci0)>0)
       xp[index_xp]=(-Bi0[index_xp]-np.sqrt(Bi0[index_xp]**2-4*Ai0*Ci0[index_xp]))/2/Ai0
      
      index_01=np.where((xp>x0*(1-lp*1e-3))  & (xp<x0+lp+lp*1e-3))
      Rix=Rix[index_01]
      Riy=Riy[index_01]
      Riz=Riz[index_01]
      xp=xp[index_01]

      Ai=(n_ip[1]**2)/(n_ip[0]**2)
      Bi=2*(Riy-Rix*(n_ip[1])/(n_ip[0]))*(n_ip[1])/(n_ip[0])
      Ci=Riz**2+(Riy-Rix*(n_ip[1])/(n_ip[0]))**2
      for i in range(len(xp)):
        def func_p(xp):
          fp=np.sqrt(p**2+2*p*(xp)+((4*e*e*p*d)/(e*e-1))) + Gp(xp)-np.sqrt(Ai*xp**2+Bi[i]*xp+Ci[i])
          return fp
        xp[i],_,ier,_=fsolve(func_p, xp[i],full_output=True)
        if ier!=1:
          xp[i]=0.1*x0 # To exclude points that did not converge.        
      rp=np.sqrt(p**2+2*p*(xp)+((4*e*e*p*d)/(e*e-1))) + Gp(xp) # parabola section radial positions in mm
      sin_phi_p=Riz/rp
      cos_phi_p=(Riy+(xp-Rix)*(n_ip[1])/(n_ip[0]))/rp
      phi_p=np.arctan2(sin_phi_p,cos_phi_p)
      indexi=np.where(((xp>=x0))  & (xp<=x0+lp))
      rp=rp[indexi]
      phi_p=phi_p[indexi]
      xp=xp[indexi]
    


    #######- Incident ray & reflected ray at parabola 
    n_ip=np.array([-np.cos(theta), -np.sin(theta),0])# incident ray direction
    if error=='yes' and approx=='yes':
      beta_p=np.arctan((p/(rp))+dGp(xp))
    elif error=='yes' and approx=='no':
      beta_p=np.arctan((p/(rp-Gp(xp)))+dGp(xp))
    elif error=='no':
      beta_p=np.arctan((p/(rp)))
    # slop angle of parabola 
    #normal direction of parabola
    n_pp=np.array([np.sin(beta_p),-np.cos(phi_p)*np.cos(beta_p),-np.sin(phi_p)*np.cos(beta_p)])
    # Direction of reflected ray from parabola
    n_rp=n_ip[:,np.newaxis]-2*np.dot(np.transpose(n_pp),n_ip)*n_pp
    #######-reflectivity at parabola 
    reflectivity_ang_p=np.rad2deg(np.arccos(np.dot(np.transpose(n_pp),n_ip)))-90
    #######
  

    ########- Incident beam on hyperboloid 
    n_rp_x=n_rp[0,:]
    n_rp_y=n_rp[1,:]
    n_rp_z=n_rp[2,:]
    A1=(n_rp_y**2+n_rp_z**2)/(n_rp_x**2)
    B1=2*(rp*(n_rp_y*np.cos(phi_p)+n_rp_z*np.sin(phi_p))/n_rp_x-xp*(n_rp_y**2+n_rp_z**2)/(n_rp_x**2))
    C1=rp**2 + (xp**2)*(n_rp_y**2+n_rp_z**2)/(n_rp_x**2) - 2*xp*rp*(n_rp_y*np.cos(phi_p)+n_rp_z*np.sin(phi_p))/n_rp_x
    A=A1-e**2+1
    B=B1-2*d*e**2
    C=C1-(e*d)**2
    xh=np.zeros_like(A)
    index_hx=np.where((B**2-4*A*C)>0)
    xh[index_hx]=(-B[index_hx]-np.sqrt(B[index_hx]**2-4*A[index_hx]*C[index_hx]))/2/A[index_hx]
    rh=np.sqrt((e*(xh+d))**2-xh**2)
    if error=="yes" and approx=='no':
      for i in range(len(xh)):
        def func(xh):
          f=np.sqrt((e*(xh+d))**2-xh**2) + Gh(xh) -np.sqrt(A1[i]*xh**2+B1[i]*xh+C1[i])
          return f
        xh[i]=fsolve(func, xh[i])
      rh=np.sqrt((e*(xh+d))**2-xh**2) + Gh(xh) 
    sin_phi_h=n_rp_z*(xh-xp)/rh/n_rp_x + rp*np.sin(phi_p)/rh
    cos_phi_h=n_rp_y*(xh-xp)/rh/n_rp_x + rp*np.cos(phi_p)/rh
    phi_h=np.arctan2(sin_phi_h,cos_phi_h)
    ########
    

    #######- Incident ray & reflected ray at hyperbola
    n_ih=n_rp
    if error=='yes' and approx=='yes':
       beta_h=np.arctan((((xh+d)*e**2-xh)/(rh)+dGh(xh)))
    elif error=='yes' and approx=='no':
       beta_h=np.arctan((((xh+d)*e**2-xh)/(rh-Gh(xh))+dGh(xh)))
    elif error=='no':
       beta_h=np.arctan(((xh+d)*e**2-xh)/rh)
    n_ph=np.array([np.sin(beta_h), -np.cos(beta_h)*cos_phi_h,-np.cos(beta_h)*sin_phi_h])
    n_rh=n_ih-2*n_ph*np.sum(n_ih*n_ph, axis=0)
    n_rh_x=n_rh[0,:]
    n_rh_y=n_rh[1,:]
    n_rh_z=n_rh[2,:]
    ########

    #######-reflectivity at Hyperbola
    reflectivity_ang_h=np.rad2deg(np.arccos(np.sum(n_ih*n_ph, axis=0)))-90
    #######
    

    
    #########- Incident Beam on detector
    yd=rh*cos_phi_h+(xd-xh)*n_rh_y/n_rh_x
    zd=rh*sin_phi_h+(xd-xh)*n_rh_z/n_rh_x
    #########

    # Define the dtype for the structured array
    dtype = np.dtype([ ('xp', np.float64), ('rp', np.float64), ('phi_p', np.float64),
        ('xh', np.float64), ('rh', np.float64), ('phi_h', np.float64),('zd', np.float64),('yd', np.float64),
        ('npx', np.float64),('npy', np.float64),('npz', np.float64),('nhx', np.float64),('nhy', np.float64),('nhz', np.float64),('theta_p', np.float64),('theta_h', np.float64)])
    ray_data = np.empty((0,), dtype=dtype)

    ray_data = np.append(ray_data, np.rec.fromarrays([xp,rp,phi_p,xh,rh,phi_h,zd,yd,n_rp_x,n_rp_y,n_rp_z,n_rh_x,n_rh_y,n_rh_z,reflectivity_ang_p,reflectivity_ang_h], dtype=dtype))
    
    ########- Restrict Hyperbola length 
    restrict_hyperbola_length=1
    if restrict_hyperbola_length==1:
     index=np.where(((xh>x0-lh))  & (xh<x0))
     ray_data=ray_data[index]
    ############
    return ray_data





