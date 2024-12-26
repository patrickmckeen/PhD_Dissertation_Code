#estimation plots for paper
from sat_ADCS_estimation import *
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from sat_ADCS_satellite import *
import numpy as np
import math
from scipy.integrate import odeint, solve_ivp, RK45
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.special import erfinv
import time
import dill as pickle
import copy
import os
import re
import matplotlib as mpl

def get_data(dirlist,name_core,path=""):
    opts = [(d,d.replace(name_core,'')) for d in dirlist if name_core in d]
    use = [d[0] for d in opts if re.match('^[0-9\.\ \_\-]*$',d[1])]
    with open(path+"/"+use[0]+"/data","rb") as fp:
        output = pickle.load(fp)
    return output

basepath = "thesis_test_files/est_files"
dirnames = next(os.walk(basepath), (None, None, []))[1]  # [] if no file

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'



if False:

    #crassidis test case; his vs. mine and 1 vs 10 sec time step; not close; w/ dipole
    names_1 = ["mine_real_world_10_","mine_real_world_1_","crassidis_real_world_10_","crassidis_real_world_1_"]
    legends_1 = ["Dynamics-Aware Filter (10s)","Dynamics-Aware Filter (1s)","USQUE (10s)","USQUE (1s)"]
    code_1 = ["mine_10","mine_1","usque_10","usque_1"]
    data_1 = [get_data(dirnames,n,basepath) for n in names_1]
    angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(d.est_state_hist[:,3:7]*d.state_hist[:,3:7],axis = 1),-1,1)**2.0 ) for d in data_1]
    mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.est_state_hist[j,3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data_1]
    ang_err_cov = [3*(180/np.pi)*4*np.arctan(np.stack([np.sqrt(np.diag(j[3:6,3:6])) for j in d.cov_hist])) for d in data_1]

    use_data = [(180/np.pi)*(d.est_state_hist[:,0:3]-d.state_hist[:,0:3]) for d in data_1]
    use_cov = [(180/np.pi)*np.stack([3*np.sqrt(np.diag(j[0:3,0:3])) for j in d.cov_hist]) for d in data_1]
    #
    #mrp
    sp_labels = ["MRP $$ (deg)","MRP $y$ (deg)","MRP $z$ (deg)"]
    av_sp_labels = ["$x$ axis (deg/s)","$y$ axis (deg/s)","$$ axis (deg/s)"]
    ylim = [-2,2]
    av_ylim = [-0.001,0.001]
    axs = [0,0,0]
    for i in range(4):
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle(legends_1[i] + " MRP Error 3-$\sigma$ Bounds in Case A")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r],"-b")
            axs[r].plot(data_1[i].t_hist,ang_err_cov[i][:,r],":b")
            axs[r].plot(data_1[i].t_hist,-1*ang_err_cov[i][:,r],":b")
            axs[r].set_ylabel(sp_labels[r])
            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        plt.savefig(code_1[i]+"mrp_TRMM_3sig.png")
        plt.draw()
        plt.pause(1)

        #AV by axis
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle(legends_1[i] + " Rate Error 3-$\sigma$ Bounds in Case A")
        for r in range(3):
            axs[r].plot(data_1[i].t_hist,use_data[i][:,r],"-b")
            axs[r].plot(data_1[i].t_hist[::1],use_cov[i][::1,r],":b")
            axs[r].plot(data_1[i].t_hist[::1],-1*use_cov[i][::1,r],":b")
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            # for i in range(4):
            axs[r].set_ylabel(av_sp_labels[r])
            axs[r].set_ylim(av_ylim[0],av_ylim[1])
        axs[r].set_xlabel("Time (s)")
        plt.savefig(code_1[i]+"axes_av_TRMM_3sig.png")
        plt.draw()
        plt.pause(1)


data_5 = get_data(dirnames,"thesis_6U_quat_RWMTQ_betterest_w_all_20240819",basepath)
angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(data_5.est_state_hist[:,3:7]*data_5.state_hist[:,3:7],axis = 1),-1,1)**2.0 )
mrp_err = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(data_5.est_state_hist[j,3:7]),data_5.state_hist[j,3:7]),0)/2 for j in range(data_5.state_hist.shape[0])]))

#ang diff overall
plt.figure()
ax = plt.subplot()
ax.plot(data_5.t_hist,angdiff)
# plt.show()
plt.title("Angular Error in Case G: Many Variables")
plt.xlabel("Time (s)")
plt.ylabel("Angular Error (deg)")
plt.draw()
plt.savefig("angular_error_caseg.png")
plt.draw()
plt.pause(1)

#log ang diff overall
plt.figure()
ax = plt.subplot()
ax.plot(data_5.t_hist,angdiff)
ax.set_yscale('log')
# plt.show()
# ax.legend()
plt.title("Angular Error in Case G: Many Variables")
plt.xlabel("Time (s)")
plt.ylabel("Angular Error (deg)")
plt.draw()
plt.savefig("log_angular_error_caseg.png")
plt.draw()
plt.pause(1)

#mrp combo
sp_labels = ["MRP $x$ (deg)","MRP $y$ (deg)","MRP $z$ (deg)"]
ylim = [-0.1,0.1]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]
fig.suptitle("MRP Error in Case G: Many Variables")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(3,1,r+1)
    #     axs[r].set_xlabel("Time (s)")
    axs[r].plot(data_5.t_hist,mrp_err[:,r])
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("mrp_caseg.png")
plt.draw()
plt.pause(1)



t_hist = data_5.t_hist
#AM
plot_the_thing(data_5.state_hist[:,7:10],data_5.est_state_hist[:,7:10],title = "Stored Angular Momentum in Case G: Many Variables",xlabel='Time (s)',ylabel = 'Stored Momenta (Nms)', norm=False,act_v_est = True,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title="'Real' Momentum", bbox_to_anchor=(0.4, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

estimated_legend = plt.legend(handles=[ handles[3], handles[4], handles[5]], labels = ["$x$","$y$","$z$"],
    loc="lower left", title="Estimated Momentum", bbox_to_anchor=(0.4,0)
)
plt.draw()
plt.savefig("am_caseg.png")
plt.draw()
plt.pause(1)
#


#mtm bias
plot_the_thing(data_5.state_hist[:,10:13],data_5.est_state_hist[:,10:13],title = "Magnetometer Bias in Case G: Many Variables",xlabel='Time (s)',ylabel = 'MTM Bias (nT)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =False,plot_now = True, save_name = "tmp")

handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title="'Real' MTM Bias", bbox_to_anchor=(0.5, 0.25)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

estimated_legend = plt.legend(handles=[ handles[3], handles[4], handles[5]], labels = ["$x$","$y$","$z$"],
    loc="lower left", title="Estimated MTM Bias", bbox_to_anchor=(0.5,0.25)
)
plt.draw()
plt.savefig("mb_caseg.png")
plt.draw()
plt.pause(1)
#gyro bias
plot_the_thing((180.0/np.pi)*data_5.state_hist[:,13:16],(180.0/np.pi)*data_5.est_state_hist[:,13:16],title = "Gyroscope Bias in Case G: Many Variables",xlabel='Time (s)',ylabel = 'Gyro Bias (deg/s)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title="'Real' Gyro Bias", bbox_to_anchor=(0.5, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

estimated_legend = plt.legend(handles=[ handles[3], handles[4], handles[5]], labels = ["$x$","$y$","$z$"],
    loc="lower left", title="Estimated Gyro Bias", bbox_to_anchor=(0.5,0)
)
plt.draw()
plt.savefig("gb_caseg.png")
plt.draw()
plt.pause(1)
#sun bias
plot_the_thing(data_5.state_hist[:,16:19],data_5.est_state_hist[:,16:19],title = "Sun Sensor Bias in Case G: Many Variables",xlabel='Time (s)',ylabel = 'Sun Sensor Bias (unitless)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title="'Real' Sun Bias", bbox_to_anchor=(0.5, 0.25)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

estimated_legend = plt.legend(handles=[ handles[3], handles[4], handles[5]], labels = ["$x$","$y$","$z$"],
    loc="lower left", title="Estimated Sun Bias", bbox_to_anchor=(0.5,0.25)
)
plt.draw()
plt.savefig("sb_caseg.png")
plt.draw()
plt.pause(1)


#dipole
plot_the_thing(data_5.state_hist[:,19:22],data_5.est_state_hist[:,19:22],title = "Residual Dipole in Case G: Many Variables",xlabel='Time (s)',ylabel = 'Residual Dipole (Am$^2$)', norm=False,act_v_est = True,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title="'Real' Dipole", bbox_to_anchor=(0.5, 0.35)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

estimated_legend = plt.legend(handles=[ handles[3], handles[4], handles[5]], labels = ["$x$","$y$","$z$"],
    loc="lower left", title="Estimated Dipole", bbox_to_anchor=(0.5,0.35)
)
plt.draw()
plt.savefig("dipole_caseg.png")
plt.draw()
plt.pause(1)


Beci = np.stack([data_5.orb_hist[j].B for j in range(len(data_5.orb_hist))])
Bbody = np.stack([data_5.orb_hist[j].B@rot_mat(data_5.state_hist[j,3:7]) for j in range(len(data_5.orb_hist))])
Seci = np.stack([data_5.orb_hist[j].S for j in range(len(data_5.orb_hist))])
Sbody = np.stack([data_5.orb_hist[j].S@rot_mat(data_5.state_hist[j,3:7]) for j in range(len(data_5.orb_hist))])
dpest_nobb = np.stack([-np.cross(Bbody[j,:],np.cross(Bbody[j,:],data_5.est_state_hist[j,19:22]))/np.dot(Bbody[j,:],Bbody[j,:]) for j in range(data_5.est_state_hist.shape[0])])
dpreal_nobb = np.stack([-np.cross(Bbody[j,:],np.cross(Bbody[j,:],data_5.state_hist[j,19:22]))/np.dot(Bbody[j,:],Bbody[j,:]) for j in range(data_5.state_hist.shape[0])])
plot_the_thing(dpreal_nobb,dpest_nobb,title = r"Residual Dipole (w/o $\mathbf{B}_{\text{body}}$) in Case G: Many Variables",xlabel='Time (s)',ylabel = 'Residual Dipole (Am$^2$)', norm=False,act_v_est = True,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title="'Real' Dipole", bbox_to_anchor=(0.35, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

estimated_legend = plt.legend(handles=[ handles[3], handles[4], handles[5]], labels = ["$x$","$y$","$z$"],
    loc="lower left", title="Estimated Dipole", bbox_to_anchor=(0.35,0)
)
plt.draw()
plt.savefig("dipole_nobb_caseg.png")
plt.draw()
plt.pause(1)

#prop torque
plot_the_thing(data_5.state_hist[:,22:25],data_5.est_state_hist[:,22:25],title = "Propulsion Disturbance Torque in Case G: Many Variables",xlabel='Time (s)',ylabel = 'Propulsion Torque (Nm)', norm=False,act_v_est = True,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title="'Real' Prop Torque", bbox_to_anchor=(0.5, 0.25)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

estimated_legend = plt.legend(handles=[ handles[3], handles[4], handles[5]], labels = ["$x$","$y$","$z$"],
    loc="lower left", title="Estimated Prop Torque", bbox_to_anchor=(0.5,0.25)
)
plt.draw()
plt.savefig("proptorq_caseg.png")
plt.draw()
plt.pause(1)




#AV by axis
sp_labels = ["$x$ axis (deg/s)","$y$ axis (deg/s)","$z$ axis (deg/s)"]
ylim = [-0.01,0.01]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]
use_data = (180/np.pi)*(data_5.est_state_hist[:,0:3]-data_5.state_hist[:,0:3])
fig.suptitle("Rate Error in Case G: Many Variables")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(4,1,r+1)
    axs[r].plot(data_5.t_hist,use_data[:,r])
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("axes_av_caseg.png")
plt.draw()
plt.pause(1)


#AV norm
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
use_data = matrix_row_norm((180/np.pi)*(data_5.est_state_hist[:,0:3]-data_5.state_hist[:,0:3]))
ax.set_title("Rate Error in Case G: Many Variables")
ax.plot(data_5.t_hist,use_data)
# ax.set_ylim(ylim[0],ylim[1])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Rate Error Norm (deg/s)")
ax.set_yscale('log')
plt.draw()
plt.savefig("log_norm_av_caseg.png")
plt.draw()
plt.pause(1)





est_quat = data_5.est_state_hist[:,3:7]
real_quat = data_5.state_hist[:,3:7]
mrp_err = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(est_quat[j,:]),real_quat[j,:]),0)/2 for j in range(real_quat.shape[0])]))
cov_test = [np.diag(j[3:6,3:6]) for j in data_5.cov_hist]
ang_err_cov = (180/np.pi)*4*np.arctan(np.stack([3*np.sqrt(np.diag(j[3:6,3:6])) for j in data_5.cov_hist]))
# mrp_err_min = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(est_quat),real_quat),0)/2 for j in range(real_quat.shape[0])]))

#mrp combo
sp_labels = ["MRP $x$ (deg)","MRP $y$ (deg)","MRP $z$ (deg)"]
ylim = [-1,1]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]
fig.suptitle("MRP Error 3-$\sigma$ Bounds in Case G")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(3,1,r+1)
    #     axs[r].set_xlabel("Time (s)")
    axs[r].plot(data_5.t_hist,mrp_err[:,r],"-b")
    axs[r].plot(data_5.t_hist,ang_err_cov[:,r],":b")
    axs[r].plot(data_5.t_hist,-1*ang_err_cov[:,r],":b")
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("mrp_caseg_3sig.png")
plt.draw()
plt.pause(1)


#AV by axis
sp_labels = ["$x$ axis (deg/s)","$y$ axis (deg/s)","$z$ axis (deg/s)"]
ylim = [-0.01,0.01]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]
use_data = (180/np.pi)*(data_5.est_state_hist[:,0:3]-data_5.state_hist[:,0:3])
use_cov = (180/np.pi)*np.stack([3*np.sqrt(np.diag(j[0:3,0:3])) for j in data_5.cov_hist])
fig.suptitle("Rate Error 3-$\sigma$ Bounds in Case G")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(4,1,r+1)
    axs[r].plot(data_5.t_hist,use_data[:,r],"-b")
    axs[r].plot(data_5.t_hist[::2],use_cov[::2,r],":b")
    axs[r].plot(data_5.t_hist[::2],-1*use_cov[::2,r],":b")
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("axes_av_caseg_3sig.png")
plt.draw()
plt.pause(1)

#world-frame AV by axis
sp_labels = ["$x$ axis (deg/s)","$y$ axis (deg/s)","$z$ axis (deg/s)"]
ylim = [-0.01,0.01]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]

use_data = (180/np.pi)*(np.stack([data_5.est_state_hist[j,0:3]@rot_mat(data_5.est_state_hist[j,3:7]).T-data_5.state_hist[j,0:3]@rot_mat(data_5.state_hist[j,3:7]).T for j in range(data_5.state_hist.shape[0])]))
use_cov = (180/np.pi)*(np.stack([3*np.sqrt(np.diag(rot_mat(data_5.est_state_hist[j,3:7])@data_5.cov_hist[j][0:3,0:3]@rot_mat(data_5.est_state_hist[j,3:7]).T)) for j in range(data_5.state_hist.shape[0])]))

# use_cov = (180/np.pi)*np.stack([3*np.sqrt(np.diag(j[0:3,0:3])) for j in data_5.cov_hist])
fig.suptitle("World-Frame Rate Error 3-$\sigma$ Bounds in Case G")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(4,1,r+1)
    axs[r].plot(data_5.t_hist,use_data[:,r],"-b")
    axs[r].plot(data_5.t_hist[::2],use_cov[::2,r],":b")
    axs[r].plot(data_5.t_hist[::2],-1*use_cov[::2,r],":b")
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("axes_av_world_caseg_3sig.png")
plt.draw()
plt.pause(1)


#Bbody
plot_the_thing(Bbody,title = r"$\mathbf{B}_{\text{body}}$ in Case G: Many Variables",xlabel='Time (s)',ylabel = r'$\mathbf{B}_{\text{body}}$ (nT)', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title=r"$\mathbf{B}_{\text{body}}$", bbox_to_anchor=(0.4, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

plt.draw()
plt.savefig("Bbody_caseg.png")
plt.draw()
plt.pause(1)
#


#Bworld
plot_the_thing(Beci,title = r"$\mathbf{B}_{\text{ECI}}$ in Case G: Many Variables",xlabel='Time (s)',ylabel = r'$\mathbf{B}_{\text{ECI}}$ (nT)', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title=r"$\mathbf{B}_{\text{ECI}}$", bbox_to_anchor=(0.4, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

plt.draw()
plt.savefig("Beci_caseg.png")
plt.draw()
plt.pause(1)
#

#Sbody
plot_the_thing(Sbody,title = r"$\mathbf{S}_{\text{body}}$ in Case G: Many Variables",xlabel='Time (s)',ylabel = r'$\mathbf{S}_{\text{body}}$ ()', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title=r"$\mathbf{S}_{\text{body}}$", bbox_to_anchor=(0.4, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

plt.draw()
plt.savefig("Sbody_caseg.png")
plt.draw()
plt.pause(1)
#


#Sbody
plot_the_thing(np.stack([j.state[3:7] for j in data_5.goal_hist]),title = "Goal Quaternion in Case G: Many Variables",xlabel='Time (s)',ylabel = 'Goal Quat ()', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2],handles[3]], labels = ["$q_0$","$q_1$","$q_2$","$q_3$"],
    loc="lower right", bbox_to_anchor=(0.4, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

plt.draw()
plt.savefig("goal_quat_caseg.png")
plt.draw()
plt.pause(1)
#

#Sworld
plot_the_thing(Seci,title = r"$\mathbf{S}_{\text{ECI}}$ in Case G: Many Variables",xlabel='Time (s)',ylabel = r'$\mathbf{S}_{\text{ECI}}$ ()', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title=r"$\mathbf{S}_{\text{ECI}}$", bbox_to_anchor=(0.4, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

plt.draw()
plt.savefig("Seci_caseg.png")
plt.draw()
plt.pause(1)
#


#missing world
missing_eci = np.stack([normalize(np.cross(Beci[j,:],Seci[j,:])) for j in range(Beci.shape[0])])
plot_the_thing(missing_eci,title = r"normalized($\mathbf{B}_{\text{ECI}}\times\mathbf{S}_{\text{ECI}}$) in Case G: Many Variables",xlabel='Time (s)',ylabel = 'Missing Vec ()', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title="Missing", bbox_to_anchor=(0.4, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

plt.draw()
plt.savefig("Missing_ECI_caseg.png")
plt.draw()
plt.pause(1)
#

#missing body
missing_body = np.stack([normalize(np.cross(Bbody[j,:],Sbody[j,:])) for j in range(Beci.shape[0])])
plot_the_thing(missing_body,title = r"normalized($\mathbf{B}_{\text{body}}\times\mathbf{S}_{\text{body}}$) in Case G: Many Variables",xlabel='Time (s)',ylabel = 'Missing Vec ()', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()

real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]], labels = ["$x$","$y$","$z$"],
    loc="lower right", title="Missing", bbox_to_anchor=(0.4, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot

plt.draw()
plt.savefig("Missing_body_caseg.png")
plt.draw()
plt.pause(1)
#

#prop by axis
sp_labels = ["$x$ axis (Nm)","$y$ axis (Nm)","$z$ axis (Nm)"]
ylim = [-5e-5,5e-5]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]
use_data = data_5.est_state_hist[:,22:25]-data_5.state_hist[:,22:25]
use_cov = np.stack([3*np.sqrt(np.diag(j[21:24,21:24])) for j in data_5.cov_hist])
fig.suptitle("Propulsion Torque Error 3-$\sigma$ Bounds in Case G")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(4,1,r+1)
    axs[r].plot(data_5.t_hist,use_data[:,r],"-b")
    axs[r].plot(data_5.t_hist[::2],use_cov[::2,r],":b")
    axs[r].plot(data_5.t_hist[::2],-1*use_cov[::2,r],":b")
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("proptorq_caseg_3sig.png")
plt.draw()
plt.pause(1)


#dipole by axis
sp_labels = ["$x$ axis (Am$^2$)","$y$ axis (Am$^2$)","$z$ axis (Am$^2$)"]
ylim = [-0.5,0.5]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]
use_data = data_5.est_state_hist[:,19:22]-data_5.state_hist[:,19:22]
use_cov = np.stack([3*np.sqrt(np.diag(j[18:21,18:21])) for j in data_5.cov_hist])
fig.suptitle("Residual Dipole Error 3-$\sigma$ Bounds in Case G")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(4,1,r+1)
    axs[r].plot(data_5.t_hist,use_data[:,r],"-b")
    axs[r].plot(data_5.t_hist[::2],use_cov[::2,r],":b")
    axs[r].plot(data_5.t_hist[::2],-1*use_cov[::2,r],":b")
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("dipole_caseg_3sig.png")
plt.draw()
plt.pause(1)


#AM by axis
sp_labels = ["$x$ axis (Nms)","$y$ axis (Nms)","$z$ axis (Nms)"]
ylim = [-1e-11,1e-11]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]
use_data = data_5.est_state_hist[:,7:10]-data_5.state_hist[:,7:10]
use_cov = np.stack([3*np.sqrt(np.diag(j[6:9,6:9])) for j in data_5.cov_hist])
fig.suptitle("Stored Momentum Error 3-$\sigma$ Bounds in Case G")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(4,1,r+1)
    axs[r].plot(data_5.t_hist,use_data[:,r],"-b")
    axs[r].plot(data_5.t_hist[::2],use_cov[::2,r],":b")
    axs[r].plot(data_5.t_hist[::2],-1*use_cov[::2,r],":b")
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("am_caseg_3sig.png")
plt.draw()
plt.pause(1)


#mtm bias by axis
ind0 = 10
sp_labels = ["$x$ axis (Am$^2$)","$y$ axis (Am$^2$)","$z$ axis (Am$^2$)"]
ylim = [-1e-7,1e-7]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]
use_data = data_5.est_state_hist[:,ind0:ind0+3]-data_5.state_hist[:,ind0:ind0+3]
use_cov = np.stack([3*np.sqrt(np.diag(j[ind0-1:ind0+2,ind0-1:ind0+2])) for j in data_5.cov_hist])
fig.suptitle("MTM Bias Error 3-$\sigma$ Bounds in Case G")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(4,1,r+1)
    axs[r].plot(data_5.t_hist,use_data[:,r],"-b")
    axs[r].plot(data_5.t_hist[::2],use_cov[::2,r],":b")
    axs[r].plot(data_5.t_hist[::2],-1*use_cov[::2,r],":b")
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("mtmbias_caseg_3sig.png")
plt.draw()
plt.pause(1)

#gyro bias by axis
ind0 = 13
sp_labels = ["$x$ axis (deg/s)","$y$ axis (deg/s)","$z$ axis (deg/s)"]
ylim = [-1e-2,1e-2]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]
use_data = (data_5.est_state_hist[:,ind0:ind0+3]-data_5.state_hist[:,ind0:ind0+3])*(180.0/np.pi)
use_cov = np.stack([3*np.sqrt(np.diag(j[ind0-1:ind0+2,ind0-1:ind0+2])) for j in data_5.cov_hist])*(180.0/np.pi)
fig.suptitle("Gyro Bias Error 3-$\sigma$ Bounds in Case G")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(4,1,r+1)
    axs[r].plot(data_5.t_hist,use_data[:,r],"-b")
    axs[r].plot(data_5.t_hist[::2],use_cov[::2,r],":b")
    axs[r].plot(data_5.t_hist[::2],-1*use_cov[::2,r],":b")
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("gyrobias_caseg_3sig.png")
plt.draw()
plt.pause(1)

#sun bias by axis
ind0 = 16
sp_labels = ["$x$ axis ()","$y$ axis ()","$z$ axis ()"]
ylim = [-1e-3,1e-3]
axs = [0,0,0]
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
axs = [ax1,ax2,ax3]
use_data = data_5.est_state_hist[:,ind0:ind0+3]-data_5.state_hist[:,ind0:ind0+3]
use_cov = np.stack([3*np.sqrt(np.diag(j[ind0-1:ind0+2,ind0-1:ind0+2])) for j in data_5.cov_hist])
fig.suptitle("Sun Bias Error 3-$\sigma$ Bounds in Case G")
for r in range(3):
    # if r>0:
    #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    # else:
    #     axs[r] = plt.subplot(4,1,r+1)
    axs[r].plot(data_5.t_hist,use_data[:,r],"-b")
    axs[r].plot(data_5.t_hist[::2],use_cov[::2,r],":b")
    axs[r].plot(data_5.t_hist[::2],-1*use_cov[::2,r],":b")
    axs[r].set_ylabel(sp_labels[r])
    axs[r].set_ylim(ylim[0],ylim[1])
axs[r].set_xlabel("Time (s)")
plt.draw()
plt.savefig("sunbias_caseg_3sig.png")
plt.draw()
plt.pause(1)
