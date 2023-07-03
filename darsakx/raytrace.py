import numpy as np
import matplotlib.pyplot as plt
from .WO_SingleShell_RayTrace import*
from .Conical_SingleShell_RayTrace import*
from .Rays_Filter import*
from .PostProcessing import*
import multiprocessing as mp
import time
import sys
from tabulate import tabulate
start_time=time.time()

class rtrace: 
    def __init__(self,Radius,Focallength,Lengthpar,Lengthhyp,ShellThickness,Theta,Raydensity,Energy,xi=1,Error="no",Approx="no",NumCore=None, SurfaceType='wo'):
        Radius=np.array(Radius); Focallength=np.array(Focallength); Lengthhyp=np.array(Lengthhyp); Lengthpar=np.array(Lengthpar); ShellThickness=np.array(ShellThickness)
        Theta.sort()
        if isinstance(Energy,list):
            Energy=Energy[0]
        if np.min(np.abs(Theta)!=0):
            Theta=np.array(Theta); Theta=np.append(Theta,0); self.Theta_0_missing='yes'        
        else: self.Theta_0_missing='no'
        if isinstance(xi,int): 
            if xi==1:
             xi=np.ones_like(Radius)
        self.arg_Radius=np.argsort(Radius); Radius=Radius[self.arg_Radius]; Focallength=Focallength[self.arg_Radius]; Lengthpar=Lengthpar[self.arg_Radius]
        Lengthhyp=Lengthhyp[self.arg_Radius] ; xi=xi[self.arg_Radius]
        self.r=Radius; self.x0=Focallength; self.xi=xi; self.lp=Lengthpar; self.lh=Lengthhyp; self.theta=Theta
        self.dl=np.sqrt(1/Raydensity)*10; self.raydensity=Raydensity; self.error=Error; self.approx=Approx;self.E=Energy; self.NumCore=NumCore; self.st=ShellThickness; self.surfacetype=SurfaceType

        if isinstance(NumCore,(int,float)):
            NumCore=int(NumCore)
            if NumCore > 0:
                cpu=mp.cpu_count()
                if NumCore > cpu:
                    print("NumCore is exciding the maximum value. Set NumCore = "+str(cpu))
                    NumCore=cpu
                print("Parallel processing: no.cores is "+str(NumCore))
                self.NumCore=NumCore
        else: self.NumCore=None
        self.table()
        self.raytrace_data=self.rtrace()
        

      
        
    
    def ray_data(self,i,j):
        #print('Radius:'+str(r[j])+' mm, Theta:'+str(theta[i])+' degree')
        if self.surfacetype=="wo":
         ray_data=wo_ray_trace(self.r[j],self.x0[j],self.xi[j],self.theta[i],self.lp[j],self.lh[j],self.approx,self.error,self.dl)### Wolter Optics
        elif self.surfacetype=="co":
         ray_data=co_ray_trace(self.r[j],self.x0[j],self.xi[j],self.theta[i],self.lp[j],self.lh[j],self.approx,self.error,self.dl)### Conical Optics
        else: raise Exception("Enter Valid surface type: wo or co")
        if j>0: ray_data = ray_filter(ray_data,self.r,self.x0,self.xi,self.lp,self.lh,self.st,self.theta[i],j)### Rays filter from shadows
        return ray_data


    def rtrace(self):
        if self.NumCore is None:
           ray_data_alltheta_allr=[]
           for i in range(len(self.theta)):
                ray_data_onetheta=[]
                for j in range(len(self.r)):
                 ray_data_onetheta.append(self.ray_data(i,j))
                ray_data_alltheta_allr.append(ray_data_onetheta)
        else:
            pool = mp.Pool(processes=self.NumCore)
            ray_data_alltheta_allr=[]
            for i in range(len(self.theta)):
                param=[]
                for j in range(len(self.r)):
                    param.append([i,j])
                ray_data_onetheta=pool.starmap(self.ray_data,param)
                ray_data_alltheta_allr.append(ray_data_onetheta)
            pool.close()
            pool.join()
        print("Run Time[s] = ","{0:.3f}".format((time.time()-start_time)))
        print("Ray Trace Completed!")  
        return ray_data_alltheta_allr,self.theta,np.mean(self.x0),self.dl
    
    def psf(self,Thetaforpsf,Pixel_size,Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon="yes"):
        reflecivity_input=self.reflecivity_input_check(Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon)
        psf0(*self.raytrace_data,self.Theta_0_missing,Thetaforpsf,Pixel_size,*reflecivity_input,self.E)
    
    def effa(self,Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon="yes"):
        reflecivity_input=self.reflecivity_input_check(Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon)
        effa0(*self.raytrace_data,self.Theta_0_missing,*reflecivity_input,self.E)
    
    def vf(self,Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon="yes"):
        reflecivity_input=self.reflecivity_input_check(Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon)
        vf0(*self.raytrace_data,self.Theta_0_missing,*reflecivity_input,self.E)

    def eef(self,Percentage,Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon="yes"):
        reflecivity_input=self.reflecivity_input_check(Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon)
        eef0(*self.raytrace_data,self.Theta_0_missing,Percentage,*reflecivity_input,self.E)
        
    def gui(self,Theta0, NumRays):
        gui_cal(*self.raytrace_data,self.lp,Theta0,NumRays,self.E)
        
       
        
        
    
    def reflecivity_input_check(self,Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon):
        if IsReflectivityCon=='no':
            if isinstance(Theta_Reflectivity,list) and isinstance(Reflectivity_p,list) and isinstance(Reflectivity_p,list):
                if len(self.r)==len(Theta_Reflectivity)==len(Reflectivity_p)==len(Reflectivity_h):
                    if all(isinstance(sublist, (list,np.ndarray))  for sublist in (Theta_Reflectivity+Reflectivity_p+Reflectivity_h)):
                         Theta_Reflectivity_temp=[]; Reflectivity_p_temp=[]; Reflectivity_h_temp=[]
                         for i in range(len(self.r)):  
                            Theta_Reflectivity_temp.append(Theta_Reflectivity[self.arg_Radius[i]])
                            Reflectivity_p_temp.append(Reflectivity_p[self.arg_Radius[i]])
                            Reflectivity_h_temp.append(Reflectivity_h[self.arg_Radius[i]])
                         Theta_Reflectivity=Theta_Reflectivity_temp; Reflectivity_p=Reflectivity_p_temp; Reflectivity_h=Reflectivity_h_temp
                    else: 
                       raise Exception("sublist in Theta_Reflectivity, Reflectivity_p and Reflectivity_h are not list or numpy.ndarray")
                else: raise Exception("Theta_Reflectivity, Reflectivity_p and Reflectivity_h must be of same length as Radius")
            else: raise TypeError("Theta_Reflectivity, Reflectivity_p and Reflectivity_h must be a list when IsReflectivityCon='no'")
        elif len(Theta_Reflectivity)==len(Reflectivity_p)==len(Reflectivity_h):
            if all(isinstance(sublist, (int,float))  for sublist in (Theta_Reflectivity+Reflectivity_p+Reflectivity_h)):
                Theta_Reflectivity_temp=[]; Reflectivity_p_temp=[]; Reflectivity_h_temp=[]
                for _ in range(len(self.r)): 
                    Theta_Reflectivity_temp.append(Theta_Reflectivity); Reflectivity_p_temp.append(Reflectivity_p); Reflectivity_h_temp.append(Reflectivity_h) 
                Theta_Reflectivity=Theta_Reflectivity_temp; Reflectivity_p=Reflectivity_p_temp; Reflectivity_h=Reflectivity_h_temp
        else:  raise TypeError("Theta_Reflectivity, Reflectivity_p and Reflectivity_h must be of same length")
        return Theta_Reflectivity,Reflectivity_p,Reflectivity_h
    
    def table(self):
        surfacetype=''
        if self.surfacetype=='wo': surfacetype='Wolter-1'
        elif self.surfacetype=='co': surfacetype='Conical-Approximation'
        table = [
               ["Telescope Parameters",''],
               ["Surface type",surfacetype],
               ["Ray Density[Rays/cm2]", self.raydensity],
               ["Figure-Error",self.error],
               ["Theta[deg]", str(np.min(self.theta))+'-'+str(np.max(self.theta))]
         ]
        
        table1 = [
               ['Sr.No','Radius[mm]', 'Lp[mm]', 'Lh[mm]','f[mm]', "\u03BE", 't[mm]'],
               *zip(np.arange(1,len(self.r)+1),self.r,self.lp,self.lh,self.x0,self.xi,self.st)
         ]
        print(tabulate(table, headers='firstrow', tablefmt="fancy_outline"))
        print("Telescope Geometrical Parameters")
        print(tabulate(table1, headers='firstrow', tablefmt="fancy_outline"))

   
   



