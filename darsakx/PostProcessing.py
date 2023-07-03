import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
import ctypes

def psf0(ray_data_alltheta_allr,theta,x0,dl,Theta_0_missing,theta0,pixel_size,theta_rf,rfp,rfh,E):
    if Theta_0_missing=="yes":
        theta=theta[0:-1]
    index=np.argmin(np.abs(theta-theta0))
    
    det_psf(ray_data_alltheta_allr[index],theta[index],x0,pixel_size,theta_rf,rfp,rfh,E)


def effa0(ray_data_alltheta_allr,theta,x0,dl,Theta_0_missing,theta_rf,rfp,rfh,E): 
    
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
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size':15})
    plt.plot(theta_arcmin,eff_Area, linewidth=4,label="Energy: "+ f"{E:.2f}"+ "keV")
    plt.ylabel('Effective-Area [cm$^2$]')
    plt.xlabel("$\Theta$ [']")
    plt.legend(loc='upper right')
    plt.show(block=False)

def vf0(ray_data_alltheta_allr,theta,x0,dl,Theta_0_missing,theta_rf,rfp,rfh,E):
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
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size':15})
    if temp==1:  plt.plot(theta_arcmin[0:-1],eff_Area[0:-1]/eff_Area[-1], linewidth=4,label="Energy: "+ f"{E:.2f}"+ "keV")
    else:  
        index=np.where(theta==0)
        plt.plot(theta_arcmin,eff_Area/eff_Area[index], linewidth=4,label="Energy: "+ f"{E:.2f}"+ "keV")
    plt.ylabel('Vignetting Factor')
    plt.xlabel("$\Theta$ [']")
    plt.legend(loc='upper right')
    plt.show(block=False)

def eef0(ray_data_alltheta_allr,theta,x0,dl,Theta_0_missing,pct,theta_rf,rfp,rfh,E):
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
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size':15})
    plt.plot(theta_arcmin,EEF_arcsec, linewidth=4,label="Energy: "+ f"{E:.2f}"+ "keV")
    plt.xlabel("$\Theta$ [']")
    plt.ylabel('EEF_'+str(pct)+'% ["]')
    plt.legend(loc='upper left')
    plt.show(block=False)
        

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



def det_psf(ray_data_allr,theta,x0,pixel_size,theta_rf,rfp,rfh,E):
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

    # Create a color plot of the intensity values
    plt.rcParams.update({'font.size':20})
    fig, ax = plt.subplots()
    #ax.set_aspect(1)
    plt.imshow(intensity, cmap='viridis',norm=colors.LogNorm(), origin='lower', extent=[-pixel_size*(num_pixels_z_1+0.5)*180*3600/np.pi/x0, pixel_size*(num_pixels_z_2+0.5)*180*3600/np.pi/x0, (yd_min-0.5*pixel_size)*180*3600/np.pi/x0, (yd_max+0.5*pixel_size)*180*3600/np.pi/x0])
    plt.colorbar()
    plt.title(label="Theta: "+ f"{theta:.2f}"+"\u00b0"+ ", Energy: "+ f"{E:.2f}"+ "keV")
    plt.xlabel('$Z_{d}$ ["]')
    plt.ylabel('$y_{d}$ ["]')
    plt.show(block=False)
    return intensity



def gui_cal(ray_data_alltheta_allr,theta,x0,dl,lp,theta0,numrays,E):
    fig = plt.figure(figsize=(8, 8))
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
            xint=np.max(x0+3*lp/2)
            ax.plot([xint,xp[k], xh[k],0], [rp[k]*np.cos(phi_p[k])+(xint-xp[k])*np.sin(theta[i]), rp[k]*np.cos(phi_p[k]), rh[k]*np.cos(phi_h[k]),yd[k]], [rp[k]*np.sin(phi_p[k]),rp[k]*np.sin(phi_p[k]), rh[k]*np.sin(phi_h[k]),zd[k]],color=color)
    
    
    ax.set_box_aspect([10, 1, 1]) 
    ax.set_axis_off()
    # Function to handle keyboard events
    def on_key(event):
        # Get the current view limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        # Adjust the view based on the pressed key
        if event.key == 'x':
            ax.view_init(elev=0, azim=0)  # XY plane
            ax.set_zlim(zlim)  # Restore the original z-axis limits
        elif event.key == 'y':
            ax.view_init(elev=0, azim=90)  # YZ plane
            ax.set_xlim(xlim)  # Restore the original x-axis limits
        elif event.key == 'z':
            ax.view_init(elev=90, azim=-90)  # ZX plane
            ax.set_ylim(ylim)  # Restore the original y-axis limits

        # Redraw the plot
        fig.canvas.draw()

    # Connect the keyboard event to the function
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=False)
 
    
    
    