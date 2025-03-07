from darsakx import rtrace
import darpanx as drp
import matplotlib.pyplot as plt
from ws_surface import*
from fo_surface import*
import numpy as np

# Telescope parameters
theta=np.linspace(0,0.5,20)
r0=np.array([200])
x0=2000*np.ones_like(r0) # in mm focal length of each shell
lp=30*np.ones_like(r0) # in mm Length of parabola of each shell
lh=30*np.ones_like(r0) # in mm Length of parabola of each shell
st=np.zeros_like(r0)+2 # in mm Thickness of  each shell

if __name__ == '__main__':
   
   m=drp.Multilayer(MultilayerType="SingleLayer",SubstrateMaterial="SiO2",LayerMaterial=["Ir"],Period=300)
   Energy=[2]
   theta_rf0=np.arange(start=0,stop=10,step=0.01)
   m.get_optical_func(Theta=theta_rf0,Energy=Energy,AllOpticalFun ="yes")
   rf0=m.Ra
   rf = [rf0] * len(r0); theta_rf = [theta_rf0] * len(r0)
   fig1, ax1 = plt.subplots(); fig2, ax2 = plt.subplots() 
   
   # Wolter
   raytrace_data=rtrace(Radius=r0,Focallength=x0,Lengthpar=lp,Lengthhyp=lh,ShellThickness=st,Theta=theta,Raydensity=500,ParallelProcessingFor='theta',NumCore=18,SurfaceType="wo")
   effa,theta_ang=raytrace_data.effa(Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,Plot='no')
   eef,theta_ang=raytrace_data.eef(Percentage=50,Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,Plot='no')
   ax1.plot(theta_ang,effa,label='W')
   ax2.plot(theta_ang,eef,label='W')
   
   # Wolter-schwarzschild
   raytrace_data=rtrace(Radius=r0,Focallength=x0,Lengthpar=lp,Lengthhyp=lh,ShellThickness=st,Theta=theta,Raydensity=500,ParallelProcessingFor='theta',NumCore=18,SurfaceType="uo", Gp=[ws_Gp],dGp=[ws_d_Gp],Gh=[ws_Gh],dGh=[ws_d_Gh])
   effa,theta_ang=raytrace_data.effa(Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,Plot='no')
   eef,theta_ang=raytrace_data.eef(Percentage=50,Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,Plot='no')
   ax1.plot(theta_ang,effa,label='WS')
   ax2.plot(theta_ang,eef,label='WS')

   #Field angle optimized
   raytrace_data=rtrace(Radius=r0,Focallength=x0,Lengthpar=lp,Lengthhyp=lh,ShellThickness=st,Theta=theta,Raydensity=500,ParallelProcessingFor='theta',NumCore=18,SurfaceType="uo", Gp=[fo_Gp],dGp=[fo_d_Gp],Gh=[fo_Gh],dGh=[fo_d_Gh])
   effa,theta_ang=raytrace_data.effa(Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,Plot='no')
   eef,theta_ang=raytrace_data.eef(Percentage=50,Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,Plot='no')
   ax1.plot(theta_ang,effa,label='FO')
   ax2.plot(theta_ang,eef,label='FO')
   ax1.set_xlabel("$\Theta$ [']")
   ax1.set_ylabel('Effective-Area [cm$^2$]')
   ax1.legend()
   ax2.set_xlabel("$\Theta$ [']")
   ax2.set_ylabel('EEF_'+str(50)+'% ["]')
   ax2.legend()
   plt.show()


