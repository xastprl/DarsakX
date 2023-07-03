import numpy as np
import matplotlib.pyplot as plt

def ray_filter(ray_data,r,x0,xi,lp,lh,st,theta,j): 
 ## ray after hitting the hyperbola at a shell at r0, ray should not hit hyperbola for the shell at r0-dr0
 if (r[j]-r[j-1])<=st[j-1]:
  raise Exception("Shell thickness error: two shells are intersecting")
 r0_next=r[j-1]+st[j-1]
 beta_next=(r0_next/x0[j-1])*((1+2*xi[j-1])/(1+xi[j-1]))/2
 alpha_next=(r0_next/x0[j-1])*(xi[j-1]/(1+xi[j-1]))/2
 index=intr_linesCone(ray_data['xh'], ray_data['rh'],ray_data['phi_h'],ray_data['nhx'],ray_data['nhy'],ray_data['nhz'],x0[j-1],r0_next,beta_next,lh[j-1])
 ray_data=np.delete(ray_data,index)
 ## ray after hitting first parabola should not hit second parabola.
 index=intr_linesCone(ray_data['xp'], ray_data['rp'],ray_data['phi_p'],ray_data['npx'],ray_data['npy'],ray_data['npz'],x0[j-1]+lp[j-1],r0_next+lp[j-1]*np.tan(alpha_next),alpha_next,lp[j-1])
 ray_data=np.delete(ray_data,index)
 
  ## ray after hitting first parabola should not hit second hyperbola.
 index=intr_linesCone(ray_data['xp'], ray_data['rp'],ray_data['phi_p'],ray_data['npx'],ray_data['npy'],ray_data['npz'],x0[j-1],r0_next,beta_next,lh[j-1])
 ray_data=np.delete(ray_data,index)
 
 ## ray after hitting second parabola should not hit first parabola.
 theta=theta*np.pi/180
 index=intr_linesCone(ray_data['xp'], ray_data['rp'],ray_data['phi_p'],np.ones_like(ray_data['npx'])*np.cos(theta),np.ones_like(ray_data['npy'])*np.sin(theta),np.zeros_like(ray_data['npz']),x0[j-1]+lp[j-1],r0_next+lp[j-1]*np.tan(alpha_next),alpha_next,lp[j-1])
 ray_data=np.delete(ray_data,index)
 return ray_data


def intr_linesCone(x_i,r_i,phi_i,nx_i,ny_i,nz_i,x0_f,r0_f,beta_f,lh_f):
 y_i=r_i*np.cos(phi_i)
 z_i=r_i*np.sin(phi_i)
 m_f=np.tan(beta_f)
 c_f=r0_f-m_f*x0_f
 A=m_f**2-(ny_i**2+nz_i**2)/(nx_i**2)
 B=2*m_f*c_f+((ny_i**2+nz_i**2)/(nx_i**2))*2*x_i-2*(y_i*ny_i+z_i*nz_i)/nx_i
 C=c_f**2-y_i**2-z_i**2-((ny_i**2+nz_i**2)/(nx_i**2))*x_i**2+2*x_i*(y_i*ny_i+z_i*nz_i)/nx_i
 x_f=np.empty_like(A)
 index_f=np.where(B**2-4*A*C>0)
 x_f[index_f]=(-B[index_f]+np.sqrt(B[index_f]**2-4*A[index_f]*C[index_f]))/2/A[index_f]
 index_A=np.where((x_f >= x0_f-lh_f) & (x_f <= x0_f))
 x_f[index_f]=(-B[index_f]-np.sqrt(B[index_f]**2-4*A[index_f]*C[index_f]))/2/A[index_f]
 index_B=np.where((x_f >= x0_f-lh_f) & (x_f <= x0_f))
 return np.union1d(index_A, index_B)
 
