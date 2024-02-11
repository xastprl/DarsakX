import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar
from mpl_toolkits.mplot3d import Axes3D
import ctypes

def psf0(ray_data_alltheta_allr,theta,x0,dl,Theta_0_missing,theta0,pixel_size,theta_rf,rfp,rfh,plot):
    if Theta_0_missing=="yes":
        theta=theta[0:-1]
    index=np.argmin(np.abs(theta-theta0))
    
    intensity_data=det_psf(ray_data_alltheta_allr[index],theta[index],x0,pixel_size,theta_rf,rfp,rfh,plot)
    return intensity_data


def effa0(ray_data_alltheta_allr,theta,x0,dl,Theta_0_missing,theta_rf,rfp,rfh,plot): 
    
    if Theta_0_missing=="yes":
        theta=theta[0:-1]
    eff_Area=np.empty_like(theta)
    
    for i in range(len(theta)):
        yd=np.empty(0); zd=np.empty(0);  R=np.empty(0)
        for j in range(len(theta_rf)):
            yd_temp=ray_data_alltheta_allr[i][j]['yd']
            zd_temp=ray_data_alltheta_allr[i][j]['zd']
            theta_p_temp=ray_data_alltheta_allr[i][j]['theta_p']
            theta_h_temp=ray_data_alltheta_allr[i][j]['theta_h']
            Rf_func_p = interp1d(theta_rf[j], rfp[j])
            Rf_func_h = interp1d(theta_rf[j], rfh[j])
            R_temp=Rf_func_p(np.abs(theta_p_temp))*Rf_func_h(np.abs(theta_h_temp))
            R=np.append(R,R_temp); yd=np.append(yd,yd_temp); zd=np.append(zd,zd_temp)
        eff_Area[i]=Eff_Area(yd,zd,R)     
    eff_Area=eff_Area*dl**2/100
    theta_arcmin=theta*60
    if plot=='yes':
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size':15})
        plt.plot(theta_arcmin,eff_Area, linewidth=4)
        plt.ylabel('Effective-Area [cm$^2$]')
        plt.xlabel("$\Theta$ [']")
        plt.grid(True)
        plt.show(block=False)
    return [eff_Area, theta_arcmin]

def vf0(ray_data_alltheta_allr,theta,x0,dl,Theta_0_missing,theta_rf,rfp,rfh,plot):
    eff_Area=np.empty_like(theta)
    temp=0
    if Theta_0_missing=="yes":
        temp=1 
    for i in range(len(theta)):
        yd=np.empty(0); zd=np.empty(0);  R=np.empty(0)
        for j in range(len(theta_rf)):
            yd_temp=ray_data_alltheta_allr[i][j]['yd']
            zd_temp=ray_data_alltheta_allr[i][j]['zd']
            theta_p_temp=ray_data_alltheta_allr[i][j]['theta_p']
            theta_h_temp=ray_data_alltheta_allr[i][j]['theta_h']
            Rf_func_p = interp1d(theta_rf[j], rfp[j])
            Rf_func_h = interp1d(theta_rf[j], rfh[j])
            R_temp=Rf_func_p(np.abs(theta_p_temp))*Rf_func_h(np.abs(theta_h_temp))
            R=np.append(R,R_temp); yd=np.append(yd,yd_temp); zd=np.append(zd,zd_temp)
        eff_Area[i]=Eff_Area(yd,zd,R)    
    eff_Area=eff_Area*dl**2/100
    theta_arcmin=theta*60
    if plot=='yes':
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size':15})
        if temp==1:  
            plt.plot(theta_arcmin[0:-1],eff_Area[0:-1]/eff_Area[-1], linewidth=4)
            theta_return=theta_arcmin[0:-1]; vf_return=eff_Area[0:-1]/eff_Area[-1]
        else:  
            index=np.where(theta==0)
            plt.plot(theta_arcmin,eff_Area/eff_Area[index], linewidth=4)
            theta_return=theta_arcmin; vf_return=eff_Area/eff_Area[index]
        plt.ylabel('Vignetting Factor')
        plt.xlabel("$\Theta$ [']")
        plt.grid(True)
        plt.show(block=False)
    else:
        if temp==1:  
            theta_return=theta_arcmin[0:-1]; vf_return=eff_Area[0:-1]/eff_Area[-1]
        else:  
            index=np.where(theta==0)
            theta_return=theta_arcmin; vf_return=eff_Area/eff_Area[index]
    return[vf_return,theta_return]

def eef0(ray_data_alltheta_allr,theta,x0,dl,Theta_0_missing,pct,theta_rf,rfp,rfh,plot):
    if Theta_0_missing=="yes":
        theta=theta[0:-1]
    eef=np.empty_like(theta) 
    for i in range(len(theta)):
        yd=np.empty(0); zd=np.empty(0);  R=np.empty(0)
        for j in range(len(theta_rf)):
            yd_temp=ray_data_alltheta_allr[i][j]['yd']
            zd_temp=ray_data_alltheta_allr[i][j]['zd']
            theta_p_temp=ray_data_alltheta_allr[i][j]['theta_p']
            theta_h_temp=ray_data_alltheta_allr[i][j]['theta_h']
            Rf_func_p = interp1d(theta_rf[j], rfp[j])
            Rf_func_h = interp1d(theta_rf[j], rfh[j])
            R_temp=Rf_func_p(np.abs(theta_p_temp))*Rf_func_h(np.abs(theta_h_temp))
            R=np.append(R,R_temp); yd=np.append(yd,yd_temp); zd=np.append(zd,zd_temp)
        eef[i]=EEF(yd,zd,pct,R)
    EEF_arcsec=3600*180*eef/x0/np.pi
    theta_arcmin=theta*60
    if plot=='yes':
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size':15})
        plt.plot(theta_arcmin,EEF_arcsec, linewidth=4)
        plt.xlabel("$\Theta$ [']")
        plt.ylabel('EEF_'+str(pct)+'% ["]')
        plt.grid(True)
        plt.show(block=False)
    return [EEF_arcsec, theta_arcmin]
        
        
def det_shape0(ray_data_alltheta_allr,theta,x0,dl,Theta_0_missing,pct,theta_rf,rfp,rfh,detectorposition,plot):
    if Theta_0_missing=="yes":
        theta=theta[0:-1] 
    new_detectorposition=np.zeros_like(theta)
    yd_mean=np.zeros_like(theta); updated_EEF_arcsec=np.zeros_like(theta) ; EEF_arcsec=np.zeros_like(theta)
    def caculate_eef(new_detectorposition0, i):
        yd=np.empty(0); zd=np.empty(0);  R=np.empty(0)
        for j in range(len(theta_rf)):
            nx_temp=ray_data_alltheta_allr[i][j]['nhx']
            ny_temp=ray_data_alltheta_allr[i][j]['nhy']
            nz_temp=ray_data_alltheta_allr[i][j]['nhz']
            yd_temp=ray_data_alltheta_allr[i][j]['yd']
            zd_temp=ray_data_alltheta_allr[i][j]['zd']
            yd_temp_new=yd_temp+(new_detectorposition0-detectorposition)*ny_temp/nx_temp
            zd_temp_new=zd_temp+(new_detectorposition0-detectorposition)*nz_temp/nx_temp
            theta_p_temp=ray_data_alltheta_allr[i][j]['theta_p']
            theta_h_temp=ray_data_alltheta_allr[i][j]['theta_h']
            Rf_func_p = interp1d(theta_rf[j], rfp[j])
            Rf_func_h = interp1d(theta_rf[j], rfh[j])
            R_temp=Rf_func_p(np.abs(theta_p_temp))*Rf_func_h(np.abs(theta_h_temp))
            R=np.append(R,R_temp); yd=np.append(yd,yd_temp_new); zd=np.append(zd,zd_temp_new)
        yd_mean0=np.sum(yd*R)/np.sum(R)
        return [EEF(yd,zd,pct,R), yd_mean0]
    
    def minimize_fun(new_detectorposition0, i):
        EEF=caculate_eef(new_detectorposition0, i)[0]
        return EEF
    
    for i in range(len(theta)):
        res=minimize_scalar(minimize_fun,args=(i,),bounds=(-x0/100,x0/100), tol=1e-5)
        new_detectorposition[i]=res.x
        updated_EEF_arcsec[i]=3600*180*res.fun/x0/np.pi
        yd_mean[i]=caculate_eef(new_detectorposition[i], i)[1]
        EEF_arcsec[i]=(caculate_eef(0,i)[0])*3600*180/x0/np.pi
    yd_mean=np.concatenate((-yd_mean,yd_mean))
    new_detectorposition=np.concatenate((new_detectorposition,new_detectorposition))
    arg=np.argsort(yd_mean)
    yd_mean=yd_mean[arg]
    new_detectorposition=new_detectorposition[arg]
    theta_arcmin=theta*60
    if plot=='yes':
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 6))
        plt.rcParams.update({'font.size':15})
        ax1.plot(yd_mean,new_detectorposition, linewidth=4, label="Optimum focal surface")
        ax1.set_xlabel("$y_{d}$ [mm]")
        ax1.set_ylabel("$x_{d}$ [mm]")
        ax1.legend(loc='best')
        ax2.plot(theta_arcmin, EEF_arcsec, linewidth=4, label="flat detector")
        ax2.plot(theta_arcmin, updated_EEF_arcsec, linewidth=4, label="curve detector")
        ax2.set_xlabel("$\Theta$ [']")
        ax2.set_ylabel('EEF_'+str(pct)+'% ["]')
        ax2.legend(loc='best')
        ax1.grid(True)
        ax2.grid(True)
        plt.show(block=False)
    return [[new_detectorposition, yd_mean],[theta_arcmin,updated_EEF_arcsec]]



def EEF(yd,zd,pct,Reflectivity):
    if len(yd.ravel())>0:
        yd_mean=np.sum(yd*Reflectivity)/np.sum(Reflectivity)
        zd_mean=np.sum(zd*Reflectivity)/np.sum(Reflectivity)
        r=np.sqrt((yd-yd_mean)**2+(zd-zd_mean)**2).ravel()
        index=np.argsort(r)
        r_sorted=r[index]
        Reflectivity_sorted=Reflectivity[index]
        r_weightage=Reflectivity_sorted
        index_1=np.abs(np.cumsum(r_weightage)/np.sum(r_weightage)-pct/100).argmin()
        r_pct=r_sorted[index_1]
    else: r_pct=0
    return r_pct


def Eff_Area(yd,zd,Reflectivity):
    if len(yd.ravel())>0:
        yd_mean=np.sum(yd*Reflectivity)/np.sum(Reflectivity)
        zd_mean=np.sum(zd*Reflectivity)/np.sum(Reflectivity)
        r=np.sqrt((yd-yd_mean)**2+(zd-zd_mean)**2).ravel()
        Reflectivity=Reflectivity.ravel()
        #index=np.where(r<EEF)
        #eff_area=np.sum(Reflectivity[index])
        eff_area=np.sum(Reflectivity)
    else: eff_area=0
    return eff_area



def det_psf(ray_data_allr,theta,x0,pixel_size,theta_rf,rfp,rfh,plot):
    zd=np.empty(0); yd=np.empty(0); R=np.empty(0); 
    for i in range(len(ray_data_allr)):
        yd_temp=ray_data_allr[i]['yd']
        zd_temp=ray_data_allr[i]['zd']
        theta_p_temp=ray_data_allr[i]['theta_p']
        theta_h_temp=ray_data_allr[i]['theta_h']
        Rf_func_p = interp1d(theta_rf[i], rfp[i])
        Rf_func_h = interp1d(theta_rf[i], rfh[i])
        R_temp=Rf_func_p(np.abs(theta_p_temp))*Rf_func_h(np.abs(theta_h_temp))
        R=np.append(R,R_temp); yd=np.append(yd,yd_temp); zd=np.append(zd,zd_temp)
    pixel_size=pixel_size*1e-3
    yd_min=np.min(yd)
    zd_min=np.min(zd)
    yd_max=np.max(yd)
    zd_max=np.max(zd)
    yd=yd-yd_min
    # Determine the number of pixels in the x and y directions
    num_pixels_y = int(np.ceil((np.max(yd)) / pixel_size))
    num_pixels_z_1=np.ceil(np.abs(zd_min)/pixel_size-0.5)
    num_pixels_z_2=np.ceil(np.abs(zd_max)/pixel_size-0.5)
    num_pixels_z=int(num_pixels_z_1+num_pixels_z_2+1)
    # Create a 2D array to store the intensity values of each pixel
    intensity = np.zeros((num_pixels_y, num_pixels_z))
    
    zd=zd+pixel_size*(num_pixels_z_1+0.5)
    # Loop through each data point and add its weight to the corresponding pixel
    for i in range(len(yd)):
        pixel_y = int(np.floor(yd[i] / pixel_size))
        pixel_z = int(np.floor(zd[i] / pixel_size))
        intensity[pixel_y, pixel_z] += R[i]

    

    zmin0=-pixel_size*(num_pixels_z_1+0.5)
    zmax0=pixel_size*(num_pixels_z_2+0.5)
    ymin0=(yd_min-0.5*pixel_size)
    ymax0=(yd_max+0.5*pixel_size)
    if plot=='yes':
        plt.rcParams.update({'font.size':20})
        fig, ax = plt.subplots(figsize=(10, 6))
        #ax.set_aspect(1)
        #plt.imshow(intensity, cmap='viridis',norm=colors.LogNorm(), origin='lower', extent=[zmin0*180*3600/np.pi/x0, zmax0*180*3600/np.pi/x0, ymin0*180*3600/np.pi/x0, ymax0*180*3600/np.pi/x0])
        plt.imshow(intensity, cmap='viridis',norm=colors.LogNorm(), origin='lower', extent=[zmin0, zmax0, ymin0, ymax0])
        plt.colorbar()
        plt.title(label="Theta: "+ f"{theta:.2f}"+"$^\circ$")
        plt.xlabel('$Z_{d}$ [mm]')
        plt.ylabel('$y_{d}$ [mm]')
        plt.show(block=False)
    return [intensity, {'zmin':zmin0,'zmax':zmax0,'ymin':ymin0,'ymax':ymax0}]



    



def gui_cal(ray_data_alltheta_allr,theta,x0,dl,radius,x0_all,lp,lh,xi,xd,theta0,numrays):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d')
    fig.tight_layout(pad=0)
    i=np.argmin(np.abs(theta-theta0))
    theta=theta*np.pi/180
    cmap = plt.get_cmap('tab10')
    for j in range(len(ray_data_alltheta_allr[i])):
        color = cmap(j % cmap.N)
        rp=ray_data_alltheta_allr[i][j]['rp']
        xp=ray_data_alltheta_allr[i][j]['xp']
        phi_p=ray_data_alltheta_allr[i][j]['phi_p']
        rh=ray_data_alltheta_allr[i][j]['rh']
        xh=ray_data_alltheta_allr[i][j]['xh']
        phi_h=ray_data_alltheta_allr[i][j]['phi_h']
        yd=ray_data_alltheta_allr[i][j]['yd']
        zd=ray_data_alltheta_allr[i][j]['zd']
        if len(rp)<numrays:
            numrays=len(rp)
        rp_subset = np.random.random_integers(low=0,high=len(rp),size=numrays)
        for k in rp_subset:
            xint=np.max(x0+2*lp)
            ax.plot([xint,xp[k], xh[k],xd], [rp[k]*np.cos(phi_p[k])+(xint-xp[k])*np.sin(theta[i]), rp[k]*np.cos(phi_p[k]), rh[k]*np.cos(phi_h[k]),yd[k]], [rp[k]*np.sin(phi_p[k]),rp[k]*np.sin(phi_p[k]), rh[k]*np.sin(phi_h[k]),zd[k]],color=color)
        angle=np.linspace(0,2*np.pi,100)
        y2=radius[j]*np.cos(angle)
        z2=radius[j]*np.sin(angle)
        x2=np.ones_like(y2)*x0_all[j]
        ax.plot(x2,y2,z2, color='gray')
        ### 
        alpha=np.arctan(radius[j]/x0_all[j])/4
        theta_p=2*xi[j]*alpha/(1+xi[j])
        theta_h=2*(1+2*xi[j])*alpha/(1+xi[j])
        y1=(radius[j]+lp[j]*np.tan(theta_p))*np.cos(angle)
        z1=(radius[j]+lp[j]*np.tan(theta_p))*np.sin(angle)
        x1=np.ones_like(y2)*(x0_all[j]+lp[j])
        ax.plot(x1,y1,z1, color='gray')
        ###
        y20=(radius[j]-lh[j]*np.tan(theta_h))*np.cos(angle)
        z20=(radius[j]-lh[j]*np.tan(theta_h))*np.sin(angle)
        x20=np.ones_like(y2)*(x0_all[j]-lh[j])
        ax.plot(x20,y20,z20, color='gray')
    
    def add_coordinate_system(ax):
        ax.text(1.2*radius[-1], 0, 0, '$X$', color='b', fontsize=18, ha='center', fontweight='bold')
        ax.text(0, 1.2*radius[-1], 0, '$Y$', color='g', fontsize=18, ha='center', fontweight='bold')
        ax.text(0, 0, 1.2*radius[-1], '$Z$', color='r', fontsize=18, ha='center', fontweight='bold')

        ax.plot([0, radius[-1]], [0, 0], [0, 0], color='b', lw=2, alpha=0.6)
        ax.plot([0, 0], [0, radius[-1]], [0, 0], color='g', lw=2, alpha=0.6)
        ax.plot([0, 0], [0, 0], [0, radius[-1]], color='r', lw=2, alpha=0.6)
    
    add_coordinate_system(ax)
    ax.set_box_aspect([x0/radius[-1], 1, 1]) 
    ax.set_axis_off()
    def on_key(event):
        if event.key == 'x':
            ax.view_init(elev=0, azim=0)
        elif event.key == 'y':
            ax.view_init(elev=0, azim=-90)
        elif event.key == 'z':
            ax.view_init(elev=90, azim=0)
        fig.canvas.draw()
    ax.set_proj_type('ortho')
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=False)
 
    
    
    