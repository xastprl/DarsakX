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

class rtrace: 
    def __init__(self,Radius,Focallength,Lengthpar,Lengthhyp,ShellThickness,Theta,Raydensity,xi=1,DetectorPosition=0,Error="no",Approx="no",Gp=None,Gh=None,dGp=None,dGh=None,NumCore=None, SurfaceType='wo'):
        self.start_time=time.time()
        if Error=='yes' and (Gp==None or Gh==None or dGp==None or dGh==None):
           raise Exception("Error='yes', hence define error-functions (Gp,Gh,dGp and dGh)")
        if Approx!='no' and Error=='no':
            raise Exception("Approximation method cannot initialize as error='no'")  
        if Gp==None: Gp=np.empty_like(Radius)
        if dGp==None: dGp=np.empty_like(Radius)
        if Gh==None: Gh=np.empty_like(Radius)
        if dGh==None: dGh=np.empty_like(Radius)
        Radius=np.array(Radius); Focallength=np.array(Focallength); Lengthhyp=np.array(Lengthhyp); Lengthpar=np.array(Lengthpar); ShellThickness=np.array(ShellThickness); Gp=np.array(Gp); Gh=np.array(Gh); dGp=np.array(dGp); dGh=np.array(dGh)
        Theta.sort()
        if np.min(np.abs(Theta)!=0):
            Theta=np.array(Theta); Theta=np.append(Theta,0); self.Theta_0_missing='yes'        
        else: self.Theta_0_missing='no'
        if isinstance(xi,int): 
            if xi==1:
             xi=np.ones_like(Radius)
        self.arg_Radius=np.argsort(Radius); Radius=Radius[self.arg_Radius]; Focallength=Focallength[self.arg_Radius]; Lengthpar=Lengthpar[self.arg_Radius]
        Lengthhyp=Lengthhyp[self.arg_Radius] ; xi=xi[self.arg_Radius]; Gp=Gp[self.arg_Radius]; dGp=dGp[self.arg_Radius]; Gh=Gh[self.arg_Radius]; dGh=dGh[self.arg_Radius]
        self.r=Radius; self.x0=Focallength; self.xi=xi; self.lp=Lengthpar; self.lh=Lengthhyp; self.theta=Theta
        self.dl=np.sqrt(1/Raydensity)*10; self.raydensity=Raydensity; self.error=Error; self.approx=Approx; self.NumCore=NumCore; self.st=ShellThickness; self.surfacetype=SurfaceType
        self.detectorposition=DetectorPosition; self.Gp=Gp; self.dGp=dGp; self.Gh=Gh; self.dGh=dGh
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
        if self.surfacetype=="wo":
         ray_data=wo_ray_trace(self.r[j],self.x0[j],self.xi[j],self.theta[i],self.lp[j],self.lh[j],self.approx,self.error,self.dl,self.detectorposition,self.Gp[j],self.Gh[j],self.dGp[j],self.dGh[j])### Wolter Optics
        elif self.surfacetype=="co":
         ray_data=co_ray_trace(self.r[j],self.x0[j],self.xi[j],self.theta[i],self.lp[j],self.lh[j],self.approx,self.error,self.dl,self.detectorposition,self.Gp[j],self.Gh[j],self.dGp[j],self.dGh[j])### Conical Optics
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
        print("Run Time[s] = ","{0:.3f}".format((time.time()-self.start_time)))
        print("Ray Trace Completed!")  
        return ray_data_alltheta_allr,self.theta,np.mean(self.x0),self.dl
    
    def psf(self,Thetaforpsf,Pixel_size,Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon="yes", Plot="yes"):
        reflecivity_input=self.reflecivity_input_check(Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon)
        intensity_data=psf0(*self.raytrace_data,self.Theta_0_missing,Thetaforpsf,Pixel_size,*reflecivity_input,Plot)
        return intensity_data
    
    def effa(self,Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon="yes", Plot="yes"):
        reflecivity_input=self.reflecivity_input_check(Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon)
        data=effa0(*self.raytrace_data,self.Theta_0_missing,*reflecivity_input,Plot)
        return data
        
        
    def eef(self,Percentage,Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon="yes", Plot="yes"):
        reflecivity_input=self.reflecivity_input_check(Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon)
        data=eef0(*self.raytrace_data,self.Theta_0_missing,Percentage,*reflecivity_input,Plot)
        return data
        
    
    def vf(self,Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon="yes", Plot="yes"):
        reflecivity_input=self.reflecivity_input_check(Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon)
        data=vf0(*self.raytrace_data,self.Theta_0_missing,*reflecivity_input,Plot)
        return data
    
    def det_shape(self,Percentage,Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon="yes", Plot="yes"):
        reflecivity_input=self.reflecivity_input_check(Theta_Reflectivity,Reflectivity_p,Reflectivity_h,IsReflectivityCon)
        data=det_shape0(*self.raytrace_data,self.Theta_0_missing,Percentage,*reflecivity_input,self.detectorposition,Plot)
        return data
   
        
    def gui(self,Theta0, NumRays):
        gui_cal(*self.raytrace_data,self.r,self.x0, self.lp,self.lh,self.xi,self.detectorposition,Theta0,NumRays)
        
    def data(self):
        ray_data_alltheta_allr,theta,x0,dl=self.raytrace_data
        return [ray_data_alltheta_allr, theta]
        
        
    
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
                else: raise Exception("Theta_Reflectivity, Reflectivity_p and Reflectivity_h must be of same length as number of shell")
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
        theta_temp=self.theta
        if np.min(self.theta)==0:
            if self.Theta_0_missing=='yes':
              theta_temp=self.theta[0:-1]
        table = [
               ["Telescope Parameters",''],
               ["Surface type",surfacetype],
               ["Ray Density[Rays/cm2]", self.raydensity],
               ["Figure-Error",self.error],
               ["Approximation-Method",self.approx],
               ["Theta[deg]", str(np.min(theta_temp))+'-'+str(np.max(theta_temp))],
               ['Detector-Position[mm]',self.detectorposition]
         ]
        
        table1 = [
               ['Sr.No','Radius[mm]', 'Lp[mm]', 'Lh[mm]','f[mm]', "\u03BE", 't[mm]'],
               *zip(np.arange(1,len(self.r)+1),self.r,self.lp,self.lh,self.x0,self.xi,self.st)
         ]
        print(tabulate(table, headers='firstrow', tablefmt="fancy_outline"))
        print("Telescope Geometrical Parameters")
        print(tabulate(table1, headers='firstrow', tablefmt="fancy_outline"))

   
   



