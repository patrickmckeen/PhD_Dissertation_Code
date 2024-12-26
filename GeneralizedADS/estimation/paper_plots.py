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

basepath = "paper_test_files"
dirnames = next(os.walk(basepath), (None, None, []))[1]  # [] if no file

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'




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
sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
av_sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
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
    plt.savefig(code_1[i]+"mrp_TRMM_initially_off_3sig.png")
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
    plt.savefig(code_1[i]+"axes_av_TRMM_initially_off_3sig.png")
    plt.draw()
    plt.pause(1)


names_2 = ["close_mine_real_world_10_","close_mine_real_world_1_","close_crassidis_real_world_10_","close_crassidis_real_world_1_"]
data_2 = [get_data(dirnames,n,basepath) for n in names_2]
legends_2 = ["Dynamics-Aware Filter (10s)","Dynamics-Aware Filter (1s)","USQUE (10s)","USQUE (1s)"]
code_2 = ["mine_10","mine_1","usque_10","usque_1"]
titles_2 = ["Angular Error in Case B: TRMM with Small Initial Bias Errors Only","MRP Error in Case B: TRMM with Small Initial Bias Errors Only","Rate Error in Case B: TRMM with Small Initial Bias Errors Only"]
mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.est_state_hist[j,3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data_2]
ang_err_cov = [3*(180/np.pi)*4*np.arctan(np.stack([np.sqrt(np.diag(j[3:6,3:6])) for j in d.cov_hist])) for d in data_2]

use_data = [(180/np.pi)*(d.est_state_hist[:,0:3]-d.state_hist[:,0:3]) for d in data_2]
use_cov = [(180/np.pi)*np.stack([3*np.sqrt(np.diag(j[0:3,0:3])) for j in d.cov_hist]) for d in data_2]
#
#mrp
sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
av_sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
ylim = [-5,5]
av_ylim = [-0.001,0.001]
axs = [0,0,0]
for i in range(4):
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    fig.suptitle(legends_2[i] + " MRP Error 3-$\sigma$ Bounds in Case B")
    for r in range(3):
        # if r>0:
        #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
        # else:
        #     axs[r] = plt.subplot(3,1,r+1)
        #     axs[r].set_xlabel("Time (s)")
        axs[r].plot(data_2[i].t_hist,mrp_err[i][:,r],"-b")
        axs[r].plot(data_2[i].t_hist,ang_err_cov[i][:,r],":b")
        axs[r].plot(data_2[i].t_hist,-1*ang_err_cov[i][:,r],":b")
        axs[r].set_ylabel(sp_labels[r])
        axs[r].set_ylim(ylim[0],ylim[1])
    axs[r].set_xlabel("Time (s)")
    plt.savefig(code_2[i]+"mrp_TRMM_initially_close_3sig.png")
    plt.draw()
    plt.pause(1)

    #AV by axis
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    fig.suptitle(legends_2[i] + " Rate Error 3-$\sigma$ Bounds in Case B")
    for r in range(3):
        axs[r].plot(data_2[i].t_hist,use_data[i][:,r],"-b")
        axs[r].plot(data_2[i].t_hist[::1],use_cov[i][::1,r],":b")
        axs[r].plot(data_2[i].t_hist[::1],-1*use_cov[i][::1,r],":b")
        # if r>0:
        #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
        # else:
        #     axs[r] = plt.subplot(4,1,r+1)
        # for i in range(4):
        axs[r].set_ylabel(av_sp_labels[r])
        axs[r].set_ylim(av_ylim[0],av_ylim[1])
    axs[r].set_xlabel("Time (s)")
    plt.savefig(code_2[i]+"axes_av_TRMM_initially_close_3sig.png")
    plt.draw()
    plt.pause(1)


if False:



    #crassidis test case; his vs. mine and 1 vs 10 sec time step; not close; w/ dipole
    names_1 = ["mine_real_world_10_","mine_real_world_1_","crassidis_real_world_10_","crassidis_real_world_1_"]
    legends_1 = ["Dynamics-Aware Filter (10s)","Dynamics-Aware Filter (1s)","USQUE (10s)","USQUE (1s)"]
    code_1 = ["mine_10","mine_1","usque_10","usque_1"]
    data_1 = [get_data(dirnames,n,basepath) for n in names_1]
    angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(d.est_state_hist[:,3:7]*d.state_hist[:,3:7],axis = 1),-1,1)**2.0 ) for d in data_1]
    mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.est_state_hist[j,3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data_1]

    #ang diff overall
    plt.figure()
    ax = plt.subplot()
    for i in range(len(names_1)):
        ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i])
        print(data_1[i].state_hist[0,3:7])
        print(quat_mult(data_1[i].state_hist[0,3:7],mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4))))
    # plt.show()
    ax.legend()
    plt.title("Angular Error in Case A: TRMM with Initial Atittude and Bias Errors")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.savefig("angular_error_TRMM_initially_off.png")
    plt.draw()
    plt.pause(1)


    #log ang diff overall
    plt.figure()
    ax = plt.subplot()
    for i in range(len(names_1)):
        ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i])
    ax.set_yscale('log')
    # plt.show()
    ax.legend()
    plt.title("Angular Error in Case A: TRMM with Initial Atittude and Bias Errors")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.savefig("log_angular_error_TRMM_initially_off.png")
    plt.draw()
    plt.pause(1)
    #
    #mrp
    sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
    ylim = [-5,5]
    for i in range(4):
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle(legends_1[i] + " MRP Error in Case A: TRMM with Larger Initial Errors")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        plt.savefig(code_1[i]+"mrp_TRMM_initially_off.png")
        plt.draw()
        plt.pause(1)

    #mrp combo
    sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
    ylim = [-5,5]
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    fig.suptitle("MRP Error in Case A: TRMM with Initial Atittude and Bias Errors")
    for r in range(3):
        # if r>0:
        #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
        # else:
        #     axs[r] = plt.subplot(3,1,r+1)
        #     axs[r].set_xlabel("Time (s)")
        for i in range(4):
            axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
        axs[r].set_ylabel(sp_labels[r])
        axs[r].set_ylim(ylim[0],ylim[1])
    axs[r].set_xlabel("Time (s)")
    axs[r].legend(legends_1)
    plt.savefig("mrp_TRMM_initially_off.png")
    plt.draw()
    plt.pause(1)

    #AV by axis
    sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
    ylim = [-0.01,0.01]
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    use_data = [(180/np.pi)*(d.est_state_hist[:,0:3]-d.state_hist[:,0:3]) for d in data_1]
    fig.suptitle("Rate Error in Case A: TRMM with Initial Atittude and Bias Errors")
    for r in range(3):
        # if r>0:
        #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
        # else:
        #     axs[r] = plt.subplot(4,1,r+1)
        for i in range(4):
            axs[r].plot(data_1[i].t_hist,use_data[i][:,r])
        axs[r].set_ylabel(sp_labels[r])
        axs[r].set_ylim(ylim[0],ylim[1])

    axs[r].set_xlabel("Time (s)")
    axs[r].legend(legends_1)
    plt.savefig("axes_av_TRMM_initially_off.png")
    plt.draw()
    plt.pause(1)


    #AV norm
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    use_data = [matrix_row_norm((180/np.pi)*(d.est_state_hist[:,0:3]-d.state_hist[:,0:3])) for d in data_1]
    fig.suptitle("Overall Rate Error in Case A: TRMM with Initial Atittude and Bias Errors")
    for i in range(4):
        ax.plot(data_1[i].t_hist,use_data[i],label = legends_1[i])
    # ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.set_ylabel("Rate Error Norm (deg/s)")
    ax.set_yscale('log')
    plt.savefig("log_norm_av_TRMM_initially_off.png")
    plt.draw()
    plt.pause(1)



    #crassidis test case; his vs. mine and 1 vs 10 sec time step; CLOSE; w/ dipole
    names_2 = ["close_mine_real_world_10_","close_mine_real_world_1_","close_crassidis_real_world_10_","close_crassidis_real_world_1_"]
    data_2 = [get_data(dirnames,n,basepath) for n in names_2]
    legends_2 = ["Dynamics-Aware Filter (10s)","Dynamics-Aware Filter (1s)","USQUE (10s)","USQUE (1s)"]
    code_2 = ["mine_10","mine_1","usque_10","usque_1"]
    titles_2 = ["Angular Error in Case B: TRMM with Small Initial Bias Errors Only","MRP Error in Case B: TRMM with Small Initial Bias Errors Only","Rate Error in Case B: TRMM with Small Initial Bias Errors Only"]
    angylim = [-5.0,5.0]
    avylim = [-0.0025,0.0025,None,0.005]

    legends = legends_2
    names = names_2
    data = data_2
    code = code_2
    titles = titles_2
    angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(d.est_state_hist[:,3:7]*d.state_hist[:,3:7],axis = 1),-1,1)**2.0 ) for d in data]
    mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.est_state_hist[j,3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data]

    #ang diff overall
    plt.figure()
    ax = plt.subplot()
    for i in range(len(names)):
        ax.plot(data[i].t_hist,angdiff[i],label=legends[i])
    # plt.show()
    ax.legend()
    plt.title(titles[0])
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.savefig("angular_error_TRMM_initially_close.png")
    plt.draw()
    plt.pause(1)

    #log ang diff overall
    plt.figure()
    ax = plt.subplot()
    for i in range(len(names)):
        ax.plot(data[i].t_hist,angdiff[i],label=legends[i])
    ax.set_yscale('log')
    # plt.show()
    ax.legend()
    plt.title(titles[0])
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.savefig("log_angular_error_TRMM_initially_close.png")
    plt.draw()
    plt.pause(1)

    #mrp combo
    sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
    ylim = angylim
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    fig.suptitle(titles[1])
    for r in range(3):
        # if r>0:
        #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
        # else:
        #     axs[r] = plt.subplot(3,1,r+1)
        #     axs[r].set_xlabel("Time (s)")
        for i in range(4):
            axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
        axs[r].set_ylabel(sp_labels[r])
        axs[r].set_ylim(ylim[0],ylim[1])
    axs[r].set_xlabel("Time (s)")
    axs[r].legend(legends)
    plt.savefig("mrp_TRMM_initially_close.png")
    plt.draw()
    plt.pause(1)


    #AV by axis
    sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
    ylim = avylim[0:2]
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    use_data = [(180/np.pi)*(d.est_state_hist[:,0:3]-d.state_hist[:,0:3]) for d in data]
    fig.suptitle(titles[2])
    for r in range(3):
        # if r>0:
        #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
        # else:
        #     axs[r] = plt.subplot(4,1,r+1)
        for i in range(4):
            axs[r].plot(data[i].t_hist,use_data[i][:,r])
        axs[r].set_ylabel(sp_labels[r])
        axs[r].set_ylim(ylim[0],ylim[1])
    axs[r].legend(legends)
    axs[r].set_xlabel("Time (s)")
    plt.savefig("axes_av_TRMM_initially_close.png")
    plt.draw()
    plt.pause(1)


    #AV norm
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    use_data = [matrix_row_norm((180/np.pi)*(d.est_state_hist[:,0:3]-d.state_hist[:,0:3])) for d in data]
    ax.set_title(titles[2])
    for i in range(4):
        ax.plot(data[i].t_hist,use_data[i],label=legends[i])
    # ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.set_ylabel("Rate Error Norm (deg/s)")
    ax.set_yscale('log')
    plt.savefig("log_norm_av_TRMM_initially_close.png")
    plt.draw()
    plt.pause(1)


    # #crassidis test case; his vs. mine and 1 vs 10 sec time step; not close; w/o dipole
    # names_3 = ["mine_real_world_no_dipole_10_","mine_real_world_no_dipole_1_","crassidis_real_world_no_dipole_10_","crassidis_real_world_no_dipole_1_"]
    # data_3 = [get_data(dirnames,n,basepath) for n in names_3]
    #
    #
    # #crassidis test case; his vs. mine and 1 vs 10 sec time step; CLOSE; w/o dipole
    # names_4 = ["close_mine_real_world_no_dipole_10_","close_mine_real_world_no_dipole_1_","close_crassidis_real_world_no_dipole_10_","close_crassidis_real_world_no_dipole_1_"]
    # data_4 = [get_data(dirnames,n,basepath) for n in names_4]

    #examples of performance in small-sat case
    #BC test case (baseline, w/o dipole); not close
    data_5 = get_data(dirnames,"baseline",basepath)
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(data_5.est_state_hist[:,3:7]*data_5.state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    mrp_err = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(data_5.est_state_hist[j,3:7]),data_5.state_hist[j,3:7]),0)/2 for j in range(data_5.state_hist.shape[0])]))

    #ang diff overall
    plt.figure()
    ax = plt.subplot()
    ax.plot(data_5.t_hist,angdiff)
    # plt.show()
    plt.title("Angular Error in Case C: CubeSat with Small Initial Bias Errors Only")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.draw()
    plt.savefig("angular_error_BC_initially_off.png")
    plt.pause(1)

    #log ang diff overall
    plt.figure()
    ax = plt.subplot()
    ax.plot(data_5.t_hist,angdiff)
    ax.set_yscale('log')
    # plt.show()
    ax.legend()
    plt.title("Angular Error in Case C: CubeSat with Initial Bias and Attitude Errors")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.draw()
    plt.savefig("log_angular_error_BC_initially_off.png")
    plt.draw()
    plt.pause(1)

    #mrp combo
    sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
    ylim = [-0.5,0.5]
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    print(data_5.state_hist[0,3:7])
    print(data_5.est_state_hist[0,3:7])
    print(data_5.state_hist[0,0:3])
    fig.suptitle("MRP Error in Case C: CubeSat with Initial Bias and Attitude Errors")
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
    plt.savefig("mrp_BC_initially_off.png")
    plt.draw()
    plt.pause(1)


    #AV by axis
    sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
    ylim = [-0.01,0.01]
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    use_data = (180/np.pi)*(data_5.est_state_hist[:,0:3]-data_5.state_hist[:,0:3])
    fig.suptitle("Rate Error in Case C: CubeSat with Initial Bias and Attitude Errors")
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
    plt.savefig("axes_av_BC_initially_off.png")
    plt.draw()
    plt.pause(1)


    #AV norm
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    use_data = matrix_row_norm((180/np.pi)*(data_5.est_state_hist[:,0:3]-data_5.state_hist[:,0:3]))
    ax.set_title("Rate Error in Case C: CubeSat with Initial Bias and Attitude Errors")
    ax.plot(data_5.t_hist,use_data)
    # ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate Error Norm (deg/s)")
    ax.set_yscale('log')
    plt.draw()
    plt.savefig("log_norm_av_BC_initially_off.png")
    plt.draw()
    plt.pause(1)

    #BC test case (baseline, w/o dipole); CLOSE
    data_6 = get_data(dirnames,"close_baseline",basepath)
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(data_6.est_state_hist[:,3:7]*data_6.state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    mrp_err = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(data_6.est_state_hist[j,3:7]),data_6.state_hist[j,3:7]),0)/2 for j in range(data_6.state_hist.shape[0])]))

    #ang diff overall
    plt.figure()
    ax = plt.subplot()
    ax.plot(data_6.t_hist,angdiff)
    # plt.show()
    plt.title("Angular Error in Case D: CubeSat with Small Initial Bias Errors Only")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.draw()
    plt.savefig("angular_error_BC_initially_close.png")
    plt.draw()
    plt.pause(1)

    #log ang diff overall
    plt.figure()
    ax = plt.subplot()
    ax.plot(data_6.t_hist,angdiff)
    ax.set_yscale('log')
    # plt.show()
    plt.title("Angular Error in Case D: CubeSat with Small Initial Bias Errors Only")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.savefig("log_angular_error_BC_initially_close.png")
    plt.draw()
    plt.pause(1)

    #mrp combo
    sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
    ylim = angylim
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    fig.suptitle("MRP Error in Case D: CubeSat with Small Initial Bias Errors Only")
    for r in range(3):
        # if r>0:
        #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
        # else:
        #     axs[r] = plt.subplot(3,1,r+1)
        #     axs[r].set_xlabel("Time (s)")
        axs[r].plot(data_6.t_hist,mrp_err[:,r])
        axs[r].set_ylabel(sp_labels[r])
        axs[r].set_ylim(ylim[0],ylim[1])
    axs[r].set_xlabel("Time (s)")
    plt.draw()
    plt.savefig("mrp_BC_initially_close.png")
    plt.draw()
    plt.pause(1)


    #AV by axis
    sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
    ylim = [-0.01,0.01]
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    use_data = (180/np.pi)*(data_6.est_state_hist[:,0:3]-data_6.state_hist[:,0:3])
    fig.suptitle("Rate Error in Case D: CubeSat with Small Initial Bias Errors Only")
    for r in range(3):
        # if r>0:
        #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
        # else:
        #     axs[r] = plt.subplot(4,1,r+1)
        axs[r].plot(data_6.t_hist,use_data[:,r])
        axs[r].set_ylabel(sp_labels[r])
        axs[r].set_ylim(ylim[0],ylim[1])
    axs[r].set_xlabel("Time (s)")
    plt.draw()
    plt.savefig("axes_av_BC_initially_close.png")
    plt.draw()
    plt.pause(1)


    #AV norm
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    use_data = matrix_row_norm((180/np.pi)*(data_6.est_state_hist[:,0:3]-data_6.state_hist[:,0:3]))
    ax.set_title("Rate Error in Case D: CubeSat with Small Initial Bias Errors Only")
    ax.plot(data_6.t_hist,use_data)
    # ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate Error Norm (deg/s)")
    ax.set_yscale('log')
    plt.draw()
    plt.savefig("log_norm_av_BC_initially_close.png")
    plt.draw()
    plt.pause(1)

    #actuator noise bias, disturbances cases
    names_7 = ["abias_missing_1","abias_there_1"]
    data_7 = [get_data(dirnames,n,basepath) for n in names_7]
    legends_7 = ["Dynamics-Aware Filter ignoring Actuator Bias","Dynamics-Aware Filter with Actuator Bias"]
    code_7 = ["no_abias","w_abias"]
    titles_7 = ["Case E: Effect of Actuator Bias Inclusion on Angular Error"]
    # angylim = [-0.1,0.1]
    # avylim = [-0.0001,0.0001,None,0.005]

    legends = legends_7
    names = names_7
    data = data_7
    code = code_7
    titles = titles_7
    angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(d.est_state_hist[:,3:7]*d.state_hist[:,3:7],axis = 1),-1,1)**2.0 ) for d in data]
    mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.est_state_hist[j,3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data]

    plt.figure()
    ax = plt.subplot()
    for i in range(len(names)):
        ax.plot(data[i].t_hist,angdiff[i],label=legends[i])
    ax.set_yscale('log')
    # plt.show()
    ax.legend()
    plt.title(titles[0])
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.savefig("log_angular_error_BC_abias_inclusion.png")
    plt.draw()
    plt.pause(1)


    names_8 = ["anoise_missing_1","anoise_there_1"]
    data_8 = [get_data(dirnames,n,basepath) for n in names_8]
    legends_8 = ["Dynamics-Aware Filter ignoring Actuator Noise","Dynamics-Aware Filter with Actuator Noise"]
    code_8 = ["no_anoise","w_anoise"]
    titles_8 = ["Case E: Effect of Actuator Noise Inclusion on Angular Error"]
    # angylim = [-0.1,0.1]
    # avylim = [-0.0001,0.0001,None,0.005]

    legends = legends_8
    names = names_8
    data = data_8
    code = code_8
    titles = titles_8
    angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(d.est_state_hist[:,3:7]*d.state_hist[:,3:7],axis = 1),-1,1)**2.0 ) for d in data]
    mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.est_state_hist[j,3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data]

    plt.figure()
    ax = plt.subplot()
    for i in range(len(names)):
        ax.plot(data[i].t_hist,angdiff[i],label=legends[i])
    ax.set_yscale('log')
    # plt.show()
    ax.legend()
    plt.title(titles[0])
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.savefig("log_angular_error_BC_anoise_inclusion.png")
    plt.draw()
    plt.pause(1)



    names_9 = ["dist_missing_1","dist_there_1"]
    data_9 = [get_data(dirnames,n,basepath) for n in names_9]
    legends_9 = ["Dynamics-Aware Filter ignoring Disturbances","Dynamics-Aware Filter with Disturbances"]
    code_9 = ["no_dist","w_dist"]
    titles_9 = ["Case E: Effect of Disturbance Inclusion on Angular Error"]
    # angylim = [-0.1,0.1]
    # avylim = [-0.0001,0.0001,None,0.005]

    legends = legends_9
    names = names_9
    data = data_9
    code = code_9
    titles = titles_9
    angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(d.est_state_hist[:,3:7]*d.state_hist[:,3:7],axis = 1),-1,1)**2.0 ) for d in data]
    mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.est_state_hist[j,3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data]

    plt.figure()
    ax = plt.subplot()
    for i in range(len(names)):
        ax.plot(data[i].t_hist,angdiff[i],label=legends[i])
    ax.set_yscale('log')
    # plt.show()
    ax.legend()
    plt.title(titles[0])
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.savefig("log_angular_error_BC_dist_inclusion.png")
    plt.draw()
    plt.pause(1)




    #utility of modeling propulsion
    names_10 = ["prop_missing_bcprop","prop_there_bcprop"]
    data_10 = [get_data(dirnames,n,basepath) for n in names_10]
    legends_10 = ["Dynamics-Aware Filter ignoring Propulsion","Dynamics-Aware Filter with Propulsion"]
    code_10 = ["no_prop","w_prop"]
    titles_10 = ["Angular Error in Case F: CubeSat with Propulsion"]
    # angylim = [-0.1,0.1]
    # avylim = [-0.0001,0.0001,None,0.005]

    legends = legends_10
    names = names_10
    data = data_10
    code = code_10
    titles = titles_10
    angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(d.est_state_hist[:,3:7]*d.state_hist[:,3:7],axis = 1),-1,1)**2.0 ) for d in data]
    mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.est_state_hist[j,3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data]

    plt.figure()
    ax = plt.subplot()
    for i in range(len(names)):
        ax.plot(data[i].t_hist,angdiff[i],label=legends[i])
    ax.set_yscale('log')
    # plt.show()
    ax.legend()
    plt.title(titles[0])
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (deg)")
    plt.savefig("log_angular_error_BC_prop_inclusion.png")
    plt.draw()
    plt.pause(1)

    #prop by axis

    plt.figure()
    ax = plt.subplot()
    use_data = np.hstack([data[1].state_hist[:,19:22],data[1].est_state_hist[:,19:22]])
    legends = ["$x$","$y$","$z$","$x$","$y$","$z$"]
    style = ["-b","-r","-g",":b",":r",":g",]
    # breakpoint()
    for i in range(6):
        ax.plot(data[1].t_hist,use_data[:,i],style[i],label=legends[i], linewidth=1)
    # plt.show()
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()

    real_legend = plt.legend(handles=[handles[0], handles[1], handles[2]],
        loc="upper right", title="'Real' Prop Torque", bbox_to_anchor=(0.5, 1)
    )
    plt.gca().add_artist(real_legend)  # Add this legend to the plot

    estimated_legend = plt.legend(handles=[ handles[3], handles[4], handles[5]],
        loc="upper left", title="Estimated Prop Torque", bbox_to_anchor=(0.5,1)
    )


    #
    # invisible_line = mpl.lines.Line2D([0], [0], color="none", lw=0)
    #
    #
    #
    # handles, labels = ax.get_legend_handles_labels()
    #
    # # Manually reorder and group labels
    # new_handles = [ invisible_line,
    #     handles[0], handles[1], handles[2],  # Real Data
    #     invisible_line, invisible_line, # Spacer
    #     handles[3], handles[4], handles[5],  # Estimated Data
    # ]
    # new_labels = [
    #     "'Real' Prop Torque",
    #     "  $x$",
    #     "  $y$",
    #     "  $z$",
    #     "",
    #     "Estimated Prop Torque",
    #     "  $x$",
    #     "  $y$",
    #     "  $z$",
    # ]
    #
    #
    #
    # # Add the legend with manual grouping
    # plt.legend(new_handles, new_labels, loc="upper right", handlelength=2)

    plt.title("Propulsion Disturbance Torque in Case F")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque from Propulsion (Nm)")



    plt.savefig("prop_torque_BC_prop_inclusion.png")



    plt.draw()
    plt.pause(1)
    # sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
    # ylim = [-0.01,0.01]
    # axs = [0,0,0]
    # fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    # # axs = [ax1,ax2,ax3]
    # fig.suptitle("Estimated vs Actual Propulsion in Nanosatellite")
    # for r in range(3):
    #     # if r>0:
    #     #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    #     # else:
    #     #     axs[r] = plt.subplot(4,1,r+1)
    #     axs[r].plot(data.t_hist,use_data[:,r])
    #     axs[r].set_ylabel(sp_labels[r])
    #     axs[r].set_ylim(ylim[0],ylim[1])
    # axs[r].set_xlabel("Time (s)")
    # plt.draw()
    # plt.savefig("dist_BC_prop_inclusion.png")
    # plt.draw()
    # plt.pause(1)

    #
    # #ang diff overall
    # plt.figure()
    # ax = plt.subplot()
    # ax.plot(data_6.t_hist,angdiff)
    # # plt.show()
    # plt.title("Nanosatellite with Propulsion Angular Error")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Angular Error (deg)")
    # plt.draw()
    # plt.savefig("angular_error_BC_prop_inclusion.png")
    # plt.draw()
    # plt.pause(1)
    #
    # #mrp combo
    # sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
    # ylim = angylim
    # axs = [0,0,0]
    # fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    # axs = [ax1,ax2,ax3]
    # fig.suptitle("Nanosatellite with Propulsion MRP Error")
    # for r in range(3):
    #     # if r>0:
    #     #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
    #     # else:
    #     #     axs[r] = plt.subplot(3,1,r+1)
    #     #     axs[r].set_xlabel("Time (s)")
    #     axs[r].plot(data_6.t_hist,mrp_err[:,r])
    #     axs[r].set_ylabel(sp_labels[r])
    #     axs[r].set_ylim(ylim[0],ylim[1])
    # axs[r].set_xlabel("Time (s)")
    # plt.draw()
    # plt.savefig("mrp_BC_prop_inclusion.png")
    # plt.draw()
    # plt.pause(1)
    #
    #
    # #AV by axis
    # sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
    # ylim = [-0.01,0.01]
    # axs = [0,0,0]
    # fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    # axs = [ax1,ax2,ax3]
    # use_data = (180/np.pi)*(data_6.est_state_hist[:,0:3]-data_6.state_hist[:,0:3])
    # fig.suptitle("Nanosatellite with Propulsion Rate Error")
    # for r in range(3):
    #     # if r>0:
    #     #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
    #     # else:
    #     #     axs[r] = plt.subplot(4,1,r+1)
    #     axs[r].plot(data_6.t_hist,use_data[:,r])
    #     axs[r].set_ylabel(sp_labels[r])
    #     axs[r].set_ylim(ylim[0],ylim[1])
    # axs[r].set_xlabel("Time (s)")
    # plt.draw()
    # plt.savefig("axes_av_BC_prop_inclusion.png")
    # plt.draw()
    # plt.pause(1)

    #examples of performance in small-sat case with 3 sigma bounds
    #BC test case (baseline, w/o dipole); not close
    data_5 = get_data(dirnames,"baseline",basepath)
    est_quat = data_5.est_state_hist[:,3:7]
    real_quat = data_5.state_hist[:,3:7]
    mrp_err = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(est_quat[j,:]),real_quat[j,:]),0)/2 for j in range(real_quat.shape[0])]))
    cov_test = [np.diag(j[3:6,3:6]) for j in data_5.cov_hist]
    ang_err_cov = (180/np.pi)*4*np.arctan(np.stack([3*np.sqrt(np.diag(j[3:6,3:6])) for j in data_5.cov_hist]))
    # mrp_err_min = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(est_quat),real_quat),0)/2 for j in range(real_quat.shape[0])]))


    #mrp combo
    sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
    ylim = [-2.5,2.5]
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    fig.suptitle("MRP Error 3-$\sigma$ Bounds in Case C")
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
    plt.savefig("mrp_BC_initially_off_3sig.png")
    plt.draw()
    plt.pause(1)


    #AV by axis
    sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
    ylim = [-0.0025,0.0025]
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    use_data = (180/np.pi)*(data_5.est_state_hist[:,0:3]-data_5.state_hist[:,0:3])
    use_cov = (180/np.pi)*np.stack([3*np.sqrt(np.diag(j[0:3,0:3])) for j in data_5.cov_hist])
    fig.suptitle("Rate Error 3-$\sigma$ Bounds in Case C")
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
    plt.savefig("axes_av_BC_initially_off_3sig.png")
    plt.draw()
    plt.pause(1)



    #BC test case (baseline, w/o dipole); CLOSE
    data_6 = get_data(dirnames,"close_baseline",basepath)
    est_quat = data_6.est_state_hist[:,3:7]
    real_quat = data_6.state_hist[:,3:7]
    mrp_err = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(est_quat[j,:]),real_quat[j,:]),0)/2 for j in range(real_quat.shape[0])]))
    cov_test = [np.diag(j[3:6,3:6]) for j in data_5.cov_hist]
    ang_err_cov = (180/np.pi)*4*np.arctan(np.stack([3*np.sqrt(np.diag(j[3:6,3:6])) for j in data_5.cov_hist]))
    # mrp_err_min = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(est_quat),real_quat),0)/2 for j in range(real_quat.shape[0])]))


    #mrp combo
    sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
    ylim = [-1.0,1.0]
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    fig.suptitle("MRP Error 3-$\sigma$ Bounds in Case D")
    for r in range(3):
        # if r>0:
        #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
        # else:
        #     axs[r] = plt.subplot(3,1,r+1)
        #     axs[r].set_xlabel("Time (s)")
        axs[r].plot(data_6.t_hist,mrp_err[:,r],"-b")
        axs[r].plot(data_6.t_hist,ang_err_cov[:,r],":b")
        axs[r].plot(data_6.t_hist,-1*ang_err_cov[:,r],":b")
        axs[r].set_ylabel(sp_labels[r])
        axs[r].set_ylim(ylim[0],ylim[1])
    axs[r].set_xlabel("Time (s)")
    plt.draw()
    plt.savefig("mrp_BC_initially_close_3sig.png")
    plt.draw()
    plt.pause(1)


    #AV by axis
    sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
    ylim = [-0.0025,0.0025]
    axs = [0,0,0]
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    axs = [ax1,ax2,ax3]
    use_data = (180/np.pi)*(data_6.est_state_hist[:,0:3]-data_6.state_hist[:,0:3])
    use_cov = (180/np.pi)*np.stack([3*np.sqrt(np.diag(j[0:3,0:3])) for j in data_6.cov_hist])
    fig.suptitle("Rate Error 3-$\sigma$ Bounds in Case D")
    for r in range(3):
        # if r>0:
        #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
        # else:
        #     axs[r] = plt.subplot(4,1,r+1)
        axs[r].plot(data_6.t_hist,use_data[:,r],"-b")
        axs[r].plot(data_6.t_hist[::2],use_cov[::2,r],":b")
        axs[r].plot(data_6.t_hist[::2],-1*use_cov[::2,r],":b")
        axs[r].set_ylabel(sp_labels[r])
        axs[r].set_ylim(ylim[0],ylim[1])
    axs[r].set_xlabel("Time (s)")
    plt.draw()
    plt.savefig("axes_av_BC_initially_close_3sig.png")
    plt.draw()
    plt.pause(1)
