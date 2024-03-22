# DarsakX: A Python Package for Designing and Analyzing Imaging Performance of X-ray Telescopes
DarsakX is a Python package that can be used to design and analyze the imaging performance of a multi-shell X-ray telescope with an optical configuration similar to Wolter-1 optics for astronomical purposes. Additionally, it can be utilized to assess the impact of figure error on the telescope's imaging performance and to optimize the optical design to improve angular resolution for wide-field telescopes.

## Pre-requisites

The following Python modules are required for the installation of DarsakX:

```
 numpy
 scipy
 matplotlib
 tabulate
```
To calculate the mirror's reflectivity, DarsakX, by default, utilizes Darpanx. The DarpanX package can be downloaded from the GitHub link- https://github.com/biswajitmb/DarpanX.git

## Installation

The DarsakX package can be downloaded from the GitHub link-https://github.com/xastprl/DarsakX

After downloading it, navigate to the DarsakX directory, which contains the setup.py file.

To install the DarsakX package system-wide:

```
sudo python3 setup.py install
```
## Usage

Importing the Package and Dependencies.
```
from darsakx import rtrace
import darpanx as drp
import matplotlib.pyplot as plt
import numpy as np
```
Define source location and telescope geometrical parameters.

```
theta=np.linspace(0,0.5,10) # source locations in deg.
r0=np.array([250,300,350]) # Radius of each shell in mm
x0=10000*np.ones_like(r0) # Focal length of each shell in mm
lp=300*np.ones_like(r0) # Length of parabola of each shell in mm
lh=300*np.ones_like(r0) # Length of hyperbola of each shell in mm
st=np.zeros_like(r0)+5 # Thickness of each shell in mm 
```
Calculate the reflectivity versus incident angle for a specific energy value using DarpanX.

```
m=drp.Multilayer(MultilayerType="SingleLayer",SubstrateMaterial="SiO2",LayerMaterial=["Ir"],Period=300)
Energy=[6] # Incident beam energy in KeV
theta_rf=np.arange(start=0,stop=5,step=0.01) # Incident angles 
m.get_optical_func(Theta=theta_rf,Energy=Energy,AllOpticalFun ="yes")
rf=m.Ra # reflectivity
```
Geometrical ray trace:

```
raytrace_data=rtrace(Radius=r0,Focallength=x0,Lengthpar=lp,Lengthhyp=lh,ShellThickness=st,Theta=theta,Raydensity=500)
```
Output plots:

```
raytrace_data.psf(Thetaforpsf= 0.2,Pixel_size=20,Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,IsReflectivityCon='no')
raytrace_data.effa(Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,IsReflectivityCon='no')
raytrace_data.det_shape(Percentage=50,Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf)
raytrace_data.vf(Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,IsReflectivityCon='no')
raytrace_data.eef(Percentage=50,Theta_Reflectivity=theta_rf,Reflectivity_p=rf,Reflectivity_h=rf,IsReflectivityCon='no')
raytrace_data.gui(Theta0=0,NumRays=10)
plt.show()
```
More details regarding DarsakX functionalities, along with examples, are described in the DarsakX_UserManual.pdf.
