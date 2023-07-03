from raytrace import rtrace
import darpanx as drp
import matplotlib.pyplot as plt
import numpy as np

# Telescope parameters
theta=np.linspace(0,0.5,2)
#theta=np.array([0])
r=np.arange(150,400,8) # radius of each shell
#np.random.shuffle(r)
#r=1000*np.array([0.65,0.87,0.99,1.23])/2
#r=[300]
x0=7500*np.ones_like(r) # in mm focal length of each shell
lp=300*np.ones_like(r) # in mm Length of parabola of each shell
lh=300*np.ones_like(r) # in mm Length of parabola of each shell
st=np.zeros_like(r)+0.5 # in mm Thickness of  each shell


if __name__ == '__main__':
   m=drp.Multilayer(MultilayerType="SingleLayer",SubstrateMaterial="SiO2",LayerMaterial=["Ir"],Period=300)
   Energy=[5]
   theta_rf0=np.arange(start=0,stop=5,step=0.01)
   m.get_optical_func(Theta=theta_rf0,Energy=Energy,AllOpticalFun ="yes")
   rf0=m.Ra
   rf=[]
   theta_rf=[]
   for i in range(len(r)):
      theta_rf.append(theta_rf0); rf.append(rf0)
     
   raytrace_data=rtrace(Radius=r,Focallength=x0,Lengthpar=lp,Lengthhyp=lh,ShellThickness=st,Theta=theta,Energy=Energy,Raydensity=300,NumCore=3,SurfaceType="co")
   #raytrace_data.psf(Thetaforpsf= 0.4,Pixel_size=20,Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,IsReflectivityCon='no')
   raytrace_data.psf(Thetaforpsf= 0.5,Pixel_size=20,Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,IsReflectivityCon='no')
   raytrace_data.effa(Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,IsReflectivityCon='no')
   #raytrace_data.vf(Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,IsReflectivityCon='no')
   raytrace_data.eef(Percentage=50,Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,IsReflectivityCon='no')
   #raytrace_data.gui(Theta0=0.4,NumRays=20)
   plt.show()


