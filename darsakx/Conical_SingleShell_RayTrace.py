import numpy as np
from scipy.optimize import fsolve
from .ErrorFunctions import*
import matplotlib.pyplot as plt

def co_ray_trace(r0,x0,xi,theta,lp,lh,approx,error,dl):
    
    # r0 mirror radius in mm
    # x0 focal lenght in mm
    # ratio of incident angle 
    # thetha off-axis angle deg
    # lp lenght of parabola in mm
    
    
    #######- WO Parameters
    theta_p=2*r0*xi/(4*x0*(xi+1))
    theta_p=np.arctan((r0+0.5*lp*np.sqrt(x0**2+r0**2)*np.tan(theta_p)/x0)/x0)*xi/(1+xi)/2
    theta_h=theta_p*(2*xi+1)/xi
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
    Rix,Riy,Riz=(x0+lp-yyi*np.sin(theta)).ravel(), (yyi*np.cos(theta)).ravel(), zzi.ravel()
    n_ip=np.array([-np.cos(theta), -np.sin(theta),0])
    n_ip_x=n_ip[0]
    n_ip_y=n_ip[1]
    n_ip_z=n_ip[2]
    mp=np.tan(theta_p)
    cp=r0-mp*x0
    Ai=mp**2-(n_ip_y**2+n_ip_z**2)/(n_ip_x**2)
    Bi=2*mp*cp+((n_ip_y**2+n_ip_z**2)/(n_ip_x**2))*2*Rix-2*(Riy*n_ip_y+Riz*n_ip_z)/n_ip_x
    Ci=cp**2-Riy**2-Riz**2-((n_ip_y**2+n_ip_z**2)/(n_ip_x**2))*Rix**2+2*Rix*(Riy*n_ip_y+Riz*n_ip_z)/n_ip_x
    xp=np.empty_like(Bi)
    index_xp=np.where((Bi**2-4*Ai*Ci)>0)
    xp[index_xp]=(-Bi[index_xp]+np.sqrt(Bi[index_xp]**2-4*Ai*Ci[index_xp]))/2/Ai
    rp=r0+(xp-x0)*mp
    sin_phi_p=Riz/rp
    cos_phi_p=(Riy+(xp-Rix)*(n_ip[1])/(n_ip[0]))/rp
    phi_p=np.arctan2(sin_phi_p,cos_phi_p)
    indexi=np.where(((xp>=x0))  & (xp<=x0+lp))
    rp=rp[indexi]
    phi_p=phi_p[indexi]
    xp=xp[indexi]
    #######
    
    #######- Incidenet Beam on Paraboloid
    rp=r0+(xp-x0)*mp  # parabola section radial positions in mm
    #######

    #######- Incident ray & reflected ray at parabola 
    n_ip=np.array([-np.cos(theta), -np.sin(theta),0])# incident ray direction
    beta_p=theta_p # slop angle of parabola 
    #normal direction of parabola
    n_pp=np.array([np.sin(beta_p)*np.ones(len(phi_p)),-np.cos(phi_p)*np.cos(beta_p),-np.sin(phi_p)*np.cos(beta_p)])
    
    # Direction of reflected ray from parabola
    n_rp=n_ip[:,np.newaxis]-2*np.dot(np.transpose(n_pp),n_ip)*n_pp
    
    #######-reflectivity at parabola 
    reflectivity_ang_p=90-np.rad2deg(np.arccos(np.dot(np.transpose(n_pp),n_ip)))
    #######


    ########- Incident beam on hyperboloid 
    n_rp_x=n_rp[0,:]
    n_rp_y=n_rp[1,:]
    n_rp_z=n_rp[2,:]
    mh=np.tan(theta_h)
    ch=r0-mh*x0
    yp=rp*np.cos(phi_p)
    zp=rp*np.sin(phi_p)
    A=mh**2-(n_rp_y**2+n_rp_z**2)/(n_rp_x**2)
    B=2*mh*ch+((n_rp_y**2+n_rp_z**2)/(n_rp_x**2))*2*xp-2*(yp*n_rp_y+zp*n_rp_z)/n_rp_x
    C=ch**2-yp**2-zp**2-((n_rp_y**2+n_rp_z**2)/(n_rp_x**2))*xp**2+2*xp*(yp*n_rp_y+zp*n_rp_z)/n_rp_x
    xh=np.empty_like(A)
    index_hx=np.where((B**2-4*A*C)>0)
    xh[index_hx]=(-B[index_hx]+np.sqrt(B[index_hx]**2-4*A[index_hx]*C[index_hx]))/2/A[index_hx]
    rh=r0+(xh-x0)*mh
    sin_phi_h=n_rp_z*(xh-xp)/rh/n_rp_x + rp*np.sin(phi_p)/rh
    cos_phi_h=n_rp_y*(xh-xp)/rh/n_rp_x + rp*np.cos(phi_p)/rh
    phi_h=np.arctan2(sin_phi_h,cos_phi_h)
    ########
   



    #######- Incident ray & reflected ray at hyperbola
    n_ih=n_rp
    beta_h=theta_h
    n_ph=np.array([np.sin(beta_h)*np.ones(len(phi_h)), -np.cos(beta_h)*cos_phi_h,-np.cos(beta_h)*sin_phi_h])
    n_rh=n_ih-2*n_ph*np.sum(n_ih*n_ph, axis=0)
    n_rh_x=n_rh[0,:]
    n_rh_y=n_rh[1,:]
    n_rh_z=n_rh[2,:]
    ########


    #######-reflectivity at Hyperbola
    reflectivity_ang_h=90-np.rad2deg(np.arccos(np.sum(n_ih*n_ph, axis=0)))
    #######
   
    #########- Incident Beam on detector
    yd=rh*cos_phi_h-xh*n_rh_y/n_rh_x
    zd=rh*sin_phi_h-xh*n_rh_z/n_rh_x
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




