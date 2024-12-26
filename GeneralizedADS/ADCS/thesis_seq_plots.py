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

basepath = "thesis_test_files"
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


data_5 = get_data(dirnames,"thesis_6U_quat_MTQ_vargoals_20240913-143417",basepath)
angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(data_5.est_state_hist[:,3:7]*data_5.state_hist[:,3:7],axis = 1),-1,1)**2.0 )
mrp_err = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(data_5.est_state_hist[j,3:7]),data_5.state_hist[j,3:7]),0)/2 for j in range(data_5.state_hist.shape[0])]))



plot_the_thing(data_5.plan_state_hist[:,3:7],title = "Quaternion in Sequentially Planned Trajectory",xlabel='Time (s)',ylabel = 'Quaternion', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")

ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
traj_starts = np.arange(150,3600,300)
plt.gca().vlines(traj_starts,ylim[0],ylim[1],linestyles='dashed',label = "Trajectory",color = '0.8',linewidth=1.1)
# plt.gca().vlines(traj_starts,ylim[1]-0.25*(ylim[1]-ylim[0]),ylim[1],linestyles='dotted',label = "Trajectory",color = 'k')
plt.gca().set_ylim(ylim[0],ylim[1])
pos = -1
midpoint = ylim[0] + 0.02*(ylim[1]-ylim[0])
dist = 0.01*(ylim[1]-ylim[0])
traj_len = 450
prec_time = 100
for ll in traj_starts[:-1]:
    # plt.gca().arrow(ll-prec_time,midpoint + dist*pos,prec_time,0.02,color = 'k',width = 0.0005,head_width=0.01, head_length=0.01,)
    sc = plt.gca().scatter(ll-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
    plt.gca().hlines(midpoint + dist*pos,ll-prec_time,ll,linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
    plt.gca().hlines(midpoint + dist*pos,ll,ll+traj_len,linestyles='solid',label = "TrajectoryLen",color = 'k')
    pos *= -1
plt.gca().scatter(traj_starts[-1]-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
plt.gca().hlines(midpoint + dist*pos,traj_starts[-1]-prec_time,traj_starts[-1],linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
plt.gca().hlines(midpoint + dist*pos,traj_starts[-1],3600,linestyles='solid',label = "TrajectoryLen",color = 'k')
pos *= -1
plt.gca().set_xlim(xlim[0],xlim[1])
handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0], handles[1], handles[2], handles[3],handles[4],handles[8],sc,handles[7]], labels = ["$q_0$","$q_1$","$q_2$","$q_3$","Traj. Start","Traj. Span","Pre-calc. Start", "Pre-calc. Span"],
    loc="upper right", bbox_to_anchor=(1.0, 1.0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("plan_quat_plot.png", dpi=600)
plt.draw()
plt.pause(1)

Rmats = np.dstack([rot_mat(data_5.state_hist[j,3:7]) for j in range(data_5.state_hist.shape[0])])
plannedRmats = np.dstack([rot_mat(data_5.plan_state_hist[j,3:7]) for j in range(data_5.state_hist.shape[0])])
point_vec_body = np.stack([j.body_vec for j in data_5.goal_hist])
point_vec_eci = np.stack([point_vec_body[j,:]@Rmats[:,:,j].T for j in range(point_vec_body.shape[0])])
goal_vec_eci = np.stack([j.eci_vec for j in data_5.goal_hist])
plan_point_vec_eci = np.stack([point_vec_body[j,:]@plannedRmats[:,:,j].T for j in range(point_vec_body.shape[0])])
planpt_err = np.arccos(np.array([np.dot(plan_point_vec_eci[j,:],goal_vec_eci[j,:]) for j in range(point_vec_body.shape[0])]))*180.0/np.pi
planpt_err[planpt_err==90.0] = np.nan


plot_the_thing(planpt_err,title = "Angle between Pointing and Goal Vector in Sequentially Planned Trajectory",xlabel='Time (s)',ylabel = 'Pointing Error (deg)', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
traj_starts = np.arange(150,3600,300)
plt.gca().vlines(traj_starts,ylim[0],ylim[1],linestyles='dashed',label = "Trajectory",color = '0.8',linewidth=1.1)
# plt.gca().vlines(traj_starts,ylim[1]-0.25*(ylim[1]-ylim[0]),ylim[1],linestyles='dotted',label = "Trajectory",color = 'k')
plt.gca().set_ylim(ylim[0],ylim[1])
pos = -1
midpoint = ylim[0] + 0.02*(ylim[1]-ylim[0])
dist = 0.01*(ylim[1]-ylim[0])
traj_len = 450
prec_time = 100
for ll in traj_starts[:-1]:
    # plt.gca().arrow(ll-prec_time,midpoint + dist*pos,prec_time,0.02,color = 'k',width = 0.0005,head_width=0.01, head_length=0.01,)
    # sc = plt.gca().scatter(ll-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
    # plt.gca().hlines(midpoint + dist*pos,ll-prec_time,ll,linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
    plt.gca().hlines(midpoint + dist*pos,ll,ll+traj_len,linestyles='solid',label = "TrajectoryLen",color = 'k')
    pos *= -1
# plt.gca().scatter(traj_starts[-1]-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
# plt.gca().hlines(midpoint + dist*pos,traj_starts[-1]-prec_time,traj_starts[-1],linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
plt.gca().hlines(midpoint + dist*pos,traj_starts[-1],3600,linestyles='solid',label = "TrajectoryLen",color = 'k')
pos *= -1


# plt.gca().vlines([150,1100,1200,1500,1600,1900,2000,2400,2500],0,ylim[1],linestyles='dashdot',label = "Goals",color = 'k',linewidth=1.5)

rect1 = mpl.patches.Rectangle((0, 0), 150, ylim[1], color='gray', alpha=0.5)  # x=1, y=1, width=2, height=3
plt.gca().add_patch(rect1)
rect2 = mpl.patches.Rectangle((1100, 0), 100, ylim[1], color='gray', alpha=0.5)  # x=1, y=1, width=2, height=3
plt.gca().add_patch(rect2)
rect3 = mpl.patches.Rectangle((1500, 0), 100, ylim[1], color='gray', alpha=0.5)  # x=1, y=1, width=2, height=3
plt.gca().add_patch(rect3)
rect4 = mpl.patches.Rectangle((1900, 0), 100, ylim[1], color='gray', alpha=0.5)  # x=1, y=1, width=2, height=3
plt.gca().add_patch(rect4)
rect5 = mpl.patches.Rectangle((2400, 0), 100, ylim[1], color='gray', alpha=0.5)  # x=1, y=1, width=2, height=3
plt.gca().add_patch(rect5)
plt.gca().set_xlim(xlim[0],xlim[1])
handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0],handles[1],handles[4],rect1], labels = ["Pointing Error","Traj. Start","Traj. Span","No Goal"],
    loc="upper right", bbox_to_anchor=(1.0, 1.0)
)
plt.gca().annotate(r"Goal: \\$-\hat{\boldsymbol{x}}$ at $-\hat{\boldsymbol{v}}$", xy=(625, ylim[1]-0.3*(ylim[1]-ylim[0])),
    xytext=(0,-5), textcoords="offset points",
    verticalalignment="top",horizontalalignment='center',
    weight="normal",
    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
plt.gca().annotate(r"Goal: \\$+\hat{\boldsymbol{z}}$ at $-\hat{\boldsymbol{r}}$", xy=(1350, ylim[1]-0.2*(ylim[1]-ylim[0])),
    xytext=(0,-5), textcoords="offset points",
    verticalalignment="top",horizontalalignment='center',
    weight="normal",
    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
plt.gca().annotate(r"Goal: \\$+\hat{\boldsymbol{z}}$ at $+\hat{\boldsymbol{r}}$", xy=(1750, ylim[1]-0.3*(ylim[1]-ylim[0])),
    xytext=(0,-5), textcoords="offset points",
    verticalalignment="top",horizontalalignment='center',
    weight="normal",
    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
plt.gca().annotate(r"Goal:   \\$+\hat{\boldsymbol{z}}$ at $+\hat{\boldsymbol{n}}$", xy=(2200, ylim[1]-0.2*(ylim[1]-ylim[0])),
    xytext=(0,-5), textcoords="offset points",
    verticalalignment="top",horizontalalignment='center',
    weight="normal",
    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
plt.gca().annotate(r"Goal: \\$-\hat{\boldsymbol{x}}$ at $-\hat{\boldsymbol{v}}$", xy=(3050, ylim[1]-0.3*(ylim[1]-ylim[0])),
    xytext=(0,-5), textcoords="offset points",
    verticalalignment="top",horizontalalignment='center',
    weight="normal",
    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("planvecang.png", dpi=600)
plt.draw()
plt.pause(1)



plot_the_thing(planpt_err,title = "Angle between Pointing and Goal Vector in Sequentially Planned Trajectory",xlabel='Time (s)',ylabel = 'Pointing Error (deg)', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
traj_starts = np.arange(150,3600,300)
ylim = (10**(-2.5),ylim[1])
# ylim[0] = 10**(-2.5)
plt.gca().vlines(traj_starts,ylim[0],ylim[1],linestyles='dashed',label = "Trajectory",color = '0.8',linewidth=1.1)
# plt.gca().vlines(traj_starts,ylim[1]-0.25*(ylim[1]-ylim[0]),ylim[1],linestyles='dotted',label = "Trajectory",color = 'k')
plt.gca().set_ylim(ylim[0],ylim[1])
plt.gca().set_yscale('log')
pos = -1
midpoint = ylim[0] + 0.00001*(ylim[1]-ylim[0])
dist = 0.0000025*(ylim[1]-ylim[0])
traj_len = 450
prec_time = 100
for ll in traj_starts[:-1]:
    # plt.gca().arrow(ll-prec_time,midpoint + dist*pos,prec_time,0.02,color = 'k',width = 0.0005,head_width=0.01, head_length=0.01,)
    # sc = plt.gca().scatter(ll-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
    # plt.gca().hlines(midpoint + dist*pos,ll-prec_time,ll,linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
    plt.gca().hlines(midpoint + dist*pos,ll,ll+traj_len,linestyles='solid',label = "TrajectoryLen",color = 'k')
    pos *= -1
# plt.gca().scatter(traj_starts[-1]-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
# plt.gca().hlines(midpoint + dist*pos,traj_starts[-1]-prec_time,traj_starts[-1],linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
plt.gca().hlines(midpoint + dist*pos,traj_starts[-1],3600,linestyles='solid',label = "TrajectoryLen",color = 'k')
pos *= -1


# plt.gca().vlines([150,1100,1200,1500,1600,1900,2000,2400,2500],0,ylim[1],linestyles='dashdot',label = "Goals",color = 'k',linewidth=1.5)

rect1 = mpl.patches.Rectangle((0, ylim[0]), 150, ylim[1], color='gray', alpha=0.5)  # x=1, y=1, width=2, height=3
plt.gca().add_patch(rect1)
rect2 = mpl.patches.Rectangle((1100, ylim[0]), 100, ylim[1], color='gray', alpha=0.5)  # x=1, y=1, width=2, height=3
plt.gca().add_patch(rect2)
rect3 = mpl.patches.Rectangle((1500, ylim[0]), 100, ylim[1], color='gray', alpha=0.5)  # x=1, y=1, width=2, height=3
plt.gca().add_patch(rect3)
rect4 = mpl.patches.Rectangle((1900, ylim[0]), 100, ylim[1], color='gray', alpha=0.5)  # x=1, y=1, width=2, height=3
plt.gca().add_patch(rect4)
rect5 = mpl.patches.Rectangle((2400, ylim[0]), 100, ylim[1], color='gray', alpha=0.5)  # x=1, y=1, width=2, height=3
plt.gca().add_patch(rect5)
plt.gca().set_xlim(xlim[0],xlim[1])
handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0],handles[1],handles[4],rect1], labels = ["Pointing Error","Traj. Start","Traj. Span","No Goal"],
    loc="upper right", bbox_to_anchor=(1.0, 1.0)
)
plt.gca().annotate(r"Goal: \\$-\hat{\boldsymbol{x}}$ at $-\hat{\boldsymbol{v}}$", xy=(625, ylim[1]-0.96*(ylim[1]-ylim[0])),
    xytext=(0,-5), textcoords="offset points",
    verticalalignment="top",horizontalalignment='center',
    weight="normal",
    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
plt.gca().annotate(r"Goal: \\$+\hat{\boldsymbol{z}}$ at $-\hat{\boldsymbol{r}}$", xy=(1350, ylim[1]-0.9*(ylim[1]-ylim[0])),
    xytext=(0,-5), textcoords="offset points",
    verticalalignment="top",horizontalalignment='center',
    weight="normal",
    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
plt.gca().annotate(r"Goal: \\$+\hat{\boldsymbol{z}}$ at $+\hat{\boldsymbol{r}}$", xy=(1750, ylim[1]-0.96*(ylim[1]-ylim[0])),
    xytext=(0,-5), textcoords="offset points",
    verticalalignment="top",horizontalalignment='center',
    weight="normal",
    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
plt.gca().annotate(r"Goal:   \\$+\hat{\boldsymbol{z}}$ at $+\hat{\boldsymbol{n}}$", xy=(2200, ylim[1]-0.9*(ylim[1]-ylim[0])),
    xytext=(0,-5), textcoords="offset points",
    verticalalignment="top",horizontalalignment='center',
    weight="normal",
    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
plt.gca().annotate(r"Goal: \\$-\hat{\boldsymbol{x}}$ at $-\hat{\boldsymbol{v}}$", xy=(3050, ylim[1]-0.96*(ylim[1]-ylim[0])),
    xytext=(0,-5), textcoords="offset points",
    verticalalignment="top",horizontalalignment='center',
    weight="normal",
    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("_logplanvecang.png", dpi=600)
plt.draw()
plt.pause(1)


plot_the_thing(data_5.plan_state_hist[:,0:3],title = "Angular Velocity in Sequentially Planned Trajectory",xlabel='Time (s)',ylabel = 'Quaternion', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")
ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
traj_starts = np.arange(150,3600,300)
plt.gca().vlines(traj_starts,ylim[0],ylim[1],linestyles='dashed',label = "Trajectory",color = '0.8',linewidth=1.1)
# plt.gca().vlines(traj_starts,ylim[1]-0.25*(ylim[1]-ylim[0]),ylim[1],linestyles='dotted',label = "Trajectory",color = 'k')
plt.gca().set_ylim(ylim[0],ylim[1])
pos = -1
midpoint = ylim[0] + 0.02*(ylim[1]-ylim[0])
dist = 0.01*(ylim[1]-ylim[0])
traj_len = 450
prec_time = 100
for ll in traj_starts[:-1]:
    # plt.gca().arrow(ll-prec_time,midpoint + dist*pos,prec_time,0.02,color = 'k',width = 0.0005,head_width=0.01, head_length=0.01,)
    sc = plt.gca().scatter(ll-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
    plt.gca().hlines(midpoint + dist*pos,ll-prec_time,ll,linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
    plt.gca().hlines(midpoint + dist*pos,ll,ll+traj_len,linestyles='solid',label = "TrajectoryLen",color = 'k')
    pos *= -1
plt.gca().scatter(traj_starts[-1]-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
plt.gca().hlines(midpoint + dist*pos,traj_starts[-1]-prec_time,traj_starts[-1],linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
plt.gca().hlines(midpoint + dist*pos,traj_starts[-1],3600,linestyles='solid',label = "TrajectoryLen",color = 'k')
pos *= -1
plt.gca().set_xlim(xlim[0],xlim[1])
handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0], handles[1], handles[2],handles[3],handles[7],sc,handles[6]], labels = [r"$\omega_x$",r"$\omega_y$",r"$\omega_z$","Traj. Start","Traj. Span","Pre-calc. Start", "Pre-calc. Span"],
    loc="upper right", bbox_to_anchor=(1.0, 1.0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("plan_av_plot.png", dpi=600)
plt.draw()
plt.pause(1)


plot_the_thing(data_5.plan_control_hist[:,0:3],title = "MTQ Command in Sequentially Planned Trajectory",xlabel='Time (s)',ylabel = 'MTQ Command (Am$^2$)', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")


ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
traj_starts = np.arange(150,3600,300)
plt.gca().vlines(traj_starts,ylim[0],ylim[1],linestyles='dashed',label = "Trajectory",color = '0.8',linewidth=1.1)
# plt.gca().vlines(traj_starts,ylim[1]-0.25*(ylim[1]-ylim[0]),ylim[1],linestyles='dotted',label = "Trajectory",color = 'k')
plt.gca().set_ylim(ylim[0],ylim[1])
pos = -1
midpoint = ylim[0] + 0.02*(ylim[1]-ylim[0])
dist = 0.01*(ylim[1]-ylim[0])
traj_len = 450
prec_time = 100
for ll in traj_starts[:-1]:
    # plt.gca().arrow(ll-prec_time,midpoint + dist*pos,prec_time,0.02,color = 'k',width = 0.0005,head_width=0.01, head_length=0.01,)
    sc = plt.gca().scatter(ll-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
    plt.gca().hlines(midpoint + dist*pos,ll-prec_time,ll,linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
    plt.gca().hlines(midpoint + dist*pos,ll,ll+traj_len,linestyles='solid',label = "TrajectoryLen",color = 'k')
    pos *= -1
plt.gca().scatter(traj_starts[-1]-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
plt.gca().hlines(midpoint + dist*pos,traj_starts[-1]-prec_time,traj_starts[-1],linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
plt.gca().hlines(midpoint + dist*pos,traj_starts[-1],3600,linestyles='solid',label = "TrajectoryLen",color = 'k')
pos *= -1
plt.gca().set_xlim(xlim[0],xlim[1])
handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0], handles[1], handles[2],handles[3],handles[7],sc,handles[6]], labels = [r"$m_x$",r"$m_y$",r"$m_z$","Traj. Start","Traj. Span","Pre-calc. Start", "Pre-calc. Span"],
    loc="upper right", bbox_to_anchor=(1.0, 1.0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("planctrl_plot.png", dpi=600)
plt.draw()
plt.pause(1)




plot_the_thing(data_5.plan_control_hist[:,0:3],title = "MTQ Command in Sequentially Planned Trajectory",xlabel='Time (s)',ylabel = 'MTQ Command (Am$^2$)', norm=False,act_v_est = False,xdata = np.array(data_5.t_hist),save =False,plot_now = True, save_name = "tmp")


ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
traj_starts = np.arange(150,3600,300)
plt.gca().vlines(traj_starts,ylim[0],ylim[1],linestyles='dashed',label = "Trajectory",color = '0.8',linewidth=1.1)
# plt.gca().vlines(traj_starts,ylim[1]-0.25*(ylim[1]-ylim[0]),ylim[1],linestyles='dotted',label = "Trajectory",color = 'k')
plt.gca().set_ylim(ylim[0],ylim[1])
pos = -1
midpoint = ylim[0] + 0.02*(ylim[1]-ylim[0])
dist = 0.01*(ylim[1]-ylim[0])
traj_len = 450
prec_time = 100
for ll in traj_starts[:-1]:
    # plt.gca().arrow(ll-prec_time,midpoint + dist*pos,prec_time,0.02,color = 'k',width = 0.0005,head_width=0.01, head_length=0.01,)
    sc = plt.gca().scatter(ll-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
    plt.gca().hlines(midpoint + dist*pos,ll-prec_time,ll,linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
    plt.gca().hlines(midpoint + dist*pos,ll,ll+traj_len,linestyles='solid',label = "TrajectoryLen",color = 'k')
    pos *= -1
plt.gca().scatter(traj_starts[-1]-prec_time,midpoint + dist*pos,marker='o',edgecolors = 'k',s=9,linewidths=0.5,color = "1")
plt.gca().hlines(midpoint + dist*pos,traj_starts[-1]-prec_time,traj_starts[-1],linestyles='dotted',label = "precalc",color = 'k',linewidth=1)
plt.gca().hlines(midpoint + dist*pos,traj_starts[-1],3600,linestyles='solid',label = "TrajectoryLen",color = 'k')
pos *= -1
plt.gca().set_xlim(xlim[0],xlim[1])
handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0], handles[1], handles[2],handles[3],handles[7],sc,handles[6]], labels = [r"$m_x$",r"$m_y$",r"$m_z$","Traj. Start","Traj. Span","Pre-calc. Start", "Pre-calc. Span"],
    loc="upper right", bbox_to_anchor=(1.0, 1.0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("planctrl_plot.png", dpi=600)
plt.draw()
plt.pause(1)
