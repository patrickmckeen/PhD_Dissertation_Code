#estimation results for paper
from sat_ADCS_estimation import *
from sat_ADCS_control import *
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from sat_ADCS_satellite import *
from sat_ADCS_ADCS import *
from run_sim import *
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
from squaternion import Quaternion
import traceback,sys,code
from bdb import BdbQuit

with open("thesis_test_files/new_tests_marker"+time.strftime("%Y%m%d-%H%M%S")+".txt", 'w') as f:
    f.write("just a file to show when new runs of tests started.")

np.random.seed(1)
all_avcov = ((0.1*math.pi/180))**2.0
all_angcov = (0.1*math.pi/180)**2.0
all_hcov = (1e-6)**2.0
all_avcov_initial_off = ((3.0*math.pi/180))**2.0
all_angcov_initial_off = 5.0**2.0#(100.0*math.pi/180)**2.0
all_hcov_initial_off = (1e-4)**2.0
all_werrcov = 1e-14#1e-20#1e-19
all_mrperrcov = 1e-10#1e-16#1e-16
all_herrcov = 1e-24#1e-24

only_gen_avcov = ((0.1*math.pi/180)/(60))**2.0
only_gen_angcov = (0.1*math.pi/180)**2.0
only_gen_hcov = (1e-6)**2.0
only_gen_avcov_initial_off = ((3.0*math.pi/180))**2.0
only_gen_angcov_initial_off = (10.0)**2.0
only_gen_hcov_initial_off = (1e-4)**2.0
only_gen_werrcov = 1e-14
only_gen_mrperrcov = 1e-6
only_gen_herrcov = 1e-22 #no dipole

q0 = normalize(np.array([0.153,0.685,0.695,0.153]))#zeroquat#normalize(np.array([0.153,0.685,0.695,0.153]))
w0 = 2*np.array([0.01,0.01,0.001])#np.array([0.53,0.53,0.053])#/(180.0/math.pi)
h0 = -0.001*(15/1000)*np.ones(3)#np.array([-0.001*j.max_h for j in real_sat.actuators if isinstance(j,RW)])

mtm_scale = 1e0
mtq_bias_on = False
mtm_bias_on = True
gyro_bias_on = True
sun_bias_on = True
prop_torq_on = True
dipole_on = True
extra_random = True



#all sensor biases, dipole, prop torque
main_sat_rwmtq =  create_GPS_6U_sat_betterest(real=True,rand=False,mtm_scale = mtm_scale,use_dipole = dipole_on,use_mtq = True,include_mtmbias = mtm_bias_on, include_mtqbias = mtq_bias_on,include_gbias = gyro_bias_on,include_sbias = sun_bias_on,use_prop = prop_torq_on)
est_sat_w_all_rwmtq =  create_GPS_6U_sat_betterest(real=False,rand=False,use_prop = prop_torq_on, estimate_prop_torq = prop_torq_on,mtm_scale = mtm_scale,use_dipole = dipole_on, estimate_dipole = dipole_on, extra_big_est_randomness = extra_random ,use_mtq = True, include_mtqbias = mtq_bias_on,estimate_mtq_bias = mtq_bias_on,include_mtmbias = mtm_bias_on, include_gbias = gyro_bias_on,include_sbias = sun_bias_on)
main_sat_rwmtq_eclipse =  create_GPS_6U_sat_betterest(real=True,rand=False,mtm_scale = mtm_scale,use_dipole = dipole_on,use_mtq = True,include_mtmbias = mtm_bias_on, include_mtqbias = mtq_bias_on,include_gbias = gyro_bias_on,include_sbias = sun_bias_on,use_prop = prop_torq_on,care_about_eclipse = True)
est_sat_w_all_rwmtq_eclipse =  create_GPS_6U_sat_betterest(real=False,rand=False,use_prop = prop_torq_on, estimate_prop_torq = prop_torq_on,mtm_scale = mtm_scale,use_dipole = dipole_on, estimate_dipole = dipole_on, extra_big_est_randomness = extra_random ,use_mtq = True, include_mtqbias = mtq_bias_on,estimate_mtq_bias = mtq_bias_on,include_mtmbias = mtm_bias_on, include_gbias = gyro_bias_on,include_sbias = sun_bias_on,care_about_eclipse = True)

# est_sat_gen =  create_GPS_6U_sat_betterest(real=False,rand=False,include_sbias = True,estimate_sun_bias = True,  include_mtmbias = True,estimate_mtm_bias = True, use_gg = False,use_drag = False, use_SRP = False,use_dipole=True,estimate_dipole = True,use_gen=True,estimate_gen_torq=True,gen_torq_std =  1e-5, gen_mag_max = 1e-3,use_prop=False,estimate_prop_torq = False)
#tracking all disturbacnes as general
est_sat_only_gen_rwmtq =  create_GPS_6U_sat_betterest(real=False,rand=False,include_sbias = sun_bias_on,estimate_sun_bias = sun_bias_on,  include_mtmbias = mtm_bias_on,estimate_mtm_bias = mtm_bias_on, use_gg = False,use_drag = False, use_SRP = False,use_dipole=False,estimate_dipole = False,use_prop=False,estimate_prop_torq = False,use_gen=True,estimate_gen_torq=True,gen_torq_std =  (2-extra_random)*5e-6, gen_mag_max = 1e-3,mtm_scale = mtm_scale, extra_big_est_randomness = extra_random ,use_mtq = True, include_mtqbias = mtq_bias_on,estimate_mtq_bias = mtq_bias_on )
est_sat_only_gen_rwmtq_eclipse =  create_GPS_6U_sat_betterest(real=False,rand=False,include_sbias = sun_bias_on,estimate_sun_bias = sun_bias_on,  include_mtmbias = mtm_bias_on,estimate_mtm_bias = mtm_bias_on, use_gg = False,use_drag = False, use_SRP = False,use_dipole=False,estimate_dipole = False,use_prop=False,estimate_prop_torq = False,use_gen=True,estimate_gen_torq=True,gen_torq_std =  (2-extra_random)*5e-6, gen_mag_max = 1e-3,mtm_scale = mtm_scale, extra_big_est_randomness = extra_random ,use_mtq = True, include_mtqbias = mtq_bias_on,estimate_mtq_bias = mtq_bias_on,care_about_eclipse = True )

orb_file = "lovera_orb_1"
axis_RW_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.RWBDOT_WITH_EKF,0.22+200*sec2cent:GovernorMode.RW_PID},
                {0.2:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3)),0.22+1500*sec2cent:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3)),0.22+2500*sec2cent:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3))},
                {0.22:-unitvecs[0],0.22+1500*sec2cent:unitvecs[2],0.22+2500*sec2cent:-unitvecs[0]})

quat_RW_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.RWBDOT_WITH_EKF,0.22+200*sec2cent:GovernorMode.RW_PID},
                {0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1])),0.22+1500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,(2/3)*np.array([-1,1,-1])),0.22+2500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1]))},
                {0.22:-unitvecs[0],0.22+1500*sec2cent:unitvecs[2],0.22+2500*sec2cent:-unitvecs[0]})

quat_thr_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.MAGIC_BDOT_WITH_EKF,0.22+200*sec2cent:GovernorMode.WIE_MAGIC_PD},
                {0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1])),0.22+1500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,(2/3)*np.array([-1,1,-1])),0.22+2500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1]))},
                {0.22:-unitvecs[0],0.22+1500*sec2cent:unitvecs[2],0.22+2500*sec2cent:-unitvecs[0]})


quat_rwmtq_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.RWBDOT_WITH_EKF,0.22+20*sec2cent:GovernorMode.MTQ_W_RW_PD},
                {0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1])),0.2+100*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[2]+unitvecs[1])),0.2+200*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1])),0.22+1500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,(2/3)*np.array([-1,1,-1])),0.22+2500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1]))},
                {0.22:-unitvecs[0],0.22+1500*sec2cent:unitvecs[2],0.22+2500*sec2cent:-unitvecs[0]})
quat_rwmtq_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.RWBDOT_WITH_EKF,0.22+20*sec2cent:GovernorMode.MTQ_W_RW_PD_MINE},
                {0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1])),0.2+100*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[2]+unitvecs[1])),0.2+200*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1])),0.22+1500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,(2/3)*np.array([-1,1,-1])),0.22+2500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1]))},
                {0.22:-unitvecs[0],0.22+1500*sec2cent:unitvecs[2],0.22+2500*sec2cent:-unitvecs[0]})



quat_rwmtq_goals_initial_off = Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.RWBDOT_WITH_EKF,0.22+10*sec2cent:GovernorMode.MTQ_W_RW_PD},
                {0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1])),0.2+100*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[2]+unitvecs[1])),0.2+200*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1])),0.22+1500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,(2/3)*np.array([-1,1,-1])),0.22+2500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,math.sqrt(2)*(unitvecs[0]-unitvecs[1]))},
                {0.22:-unitvecs[0],0.22+1500*sec2cent:unitvecs[2],0.22+2500*sec2cent:-unitvecs[0]})



                #first and third goal is pointing -x-axis anti-ram, +z-axis Nadir. second goal is +y-axis nadir, -x-axis anti-ram. based on wisniewski orbit convention and an imagined asteria

prop_schedule = {0.22+400*sec2cent:True,0.22+1500*sec2cent:False,0.22+2800*sec2cent:True}
sens_cov0_scale = 1e-6
if mtm_bias_on:
    cov0_estimate_all_rwmtq = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,all_hcov*np.eye(3),np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*mtm_bias_on)*(3e-6)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM) and j.has_bias]),np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(1e-4)**2.0)
    cov0_estimate_all_rwmtq_nodipole = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,all_hcov*np.eye(3),np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*mtm_bias_on)*(3e-6)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM) and j.has_bias]),np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,np.eye(prop_torq_on*3)*(1e-4)**2.0)

else:
    cov0_estimate_all_rwmtq = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,all_hcov*np.eye(3),np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(1e-4)**2.0)
    cov0_estimate_all_rwmtq_nodipole = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,all_hcov*np.eye(3),np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,np.eye(prop_torq_on*3)*(1e-4)**2.0)


if mtm_bias_on:
    cov0_estimate_all_initial_off_rwmtq = block_diag(np.eye(3)*all_avcov_initial_off,np.eye(3)*all_angcov_initial_off,all_hcov_initial_off*np.eye(3),np.eye(mtq_bias_on*3)*0.25**2.0,sens_cov0_scale*np.eye(3*mtm_bias_on)*(3e-6)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM) and j.has_bias]),sens_cov0_scale*np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,sens_cov0_scale*np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(1e-4)**2.0)
else:
    cov0_estimate_all_initial_off_rwmtq = block_diag(np.eye(3)*all_avcov_initial_off,np.eye(3)*all_angcov_initial_off,all_hcov_initial_off*np.eye(3),np.eye(mtq_bias_on*3)*0.25**2.0,sens_cov0_scale*np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,sens_cov0_scale*np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(1e-4)**2.0)


base_cov = block_diag(np.eye(3)*simple_avcov,np.eye(3)*simple_angcov,np.eye(3)*simple_hcov)
base_cov_initial_off = block_diag(np.eye(3)*simple_avcov_initial_off,np.eye(3)*simple_angcov_initial_off,np.eye(3)*simple_hcov_initial_off)
# cov0_estimate_gen = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,np.eye(3)*all_hcov,np.eye(3)*(1e-7)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM)]),np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,0.15*np.eye(3),np.eye(3)*(1e-4)**2.0)
cov0_estimate_only_gen_rwmtq = block_diag(base_cov,np.eye(3)*(5e-7)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM)]),np.eye(3)*(0.2*((math.pi/180.0)))**2.0,np.eye(3)*(0.3*0.2)**2.0,np.eye(3)*(1e-4)**2.0)
cov0_estimate_only_gen_initial_off_rwmtq = block_diag(base_cov_initial_off,sens_cov0_scale*np.eye(3)*(5e-7)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM)]),sens_cov0_scale*np.eye(3)*(0.2*((math.pi/180.0)))**2.0,sens_cov0_scale*np.eye(3)*(0.3*0.2)**2.0,np.eye(3)*(1e-4)**2.0)

dt = 1
mini_dist_ic = np.eye(0)# block_diag(*[j.std**2.0 for j in mini_est_sat_w_all.disturbances if j.estimated_param])
if prop_torq_on or dipole_on:
    all_dist_ic_rwmtq = block_diag(*[j.std**2.0 for j in est_sat_w_all_rwmtq.disturbances if j.estimated_param])
else:
    all_dist_ic_rwmtq = np.eye(0)
only_gen_dist_ic_rwmtq = block_diag(*[j.std**2.0 for j in est_sat_only_gen_rwmtq.disturbances if j.estimated_param])



est_sat = est_sat_w_all_rwmtq
all_int_cov_rwmtq =  dt*block_diag(np.block([[np.eye(3)*all_werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*all_mrperrcov]]),dt*np.eye(3)*all_herrcov)#,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),all_dist_ic_rwmtq)
est_sat = est_sat_only_gen_rwmtq
only_gen_int_cov_rwmtq =  dt*block_diag(np.block([[np.eye(3)*only_gen_werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*only_gen_mrperrcov]]),dt*np.eye(3)*only_gen_herrcov)#,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),only_gen_dist_ic_rwmtq)

case_quat_RWMTQ_all = ["thesis_6U_quat_RWMTQ_est_w_all",           est_sat_w_all_rwmtq,     main_sat_rwmtq,    1,  w0,      q0,h0,      cov0_estimate_all_rwmtq,      True,  quat_rwmtq_goals, prop_schedule,      4500,    orb_file,all_int_cov_rwmtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e0,None,"",0.0,1.0,w0,q0,h0]
case_quat_RWMTQ_only_gen = ["thesis_6U_quat_RWMTQ_est_w_only_gen",           est_sat_only_gen_rwmtq,       main_sat_rwmtq, 1,  w0,      q0,h0,      cov0_estimate_only_gen_rwmtq,      True,  quat_rwmtq_goals, prop_schedule,      4500,    orb_file,only_gen_int_cov_rwmtq,9,3,0,1e-3,0,"",0.0,1.0,w0,q0,h0]


case_quat_RWMTQ_all_initial_off = ["thesis_6U_quat_RWMTQ_est_w_all_initial_off",           est_sat_w_all_rwmtq,     main_sat_rwmtq,    1,   w0,q0,h0,      cov0_estimate_all_initial_off_rwmtq,      True,  quat_rwmtq_goals, prop_schedule,      4500,    orb_file,all_int_cov_rwmtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,0*w0,unitvecs4[0],0*h0]
case_quat_RWMTQ_only_gen_initial_off = ["thesis_6U_quat_RWMTQ_est_w_only_gen_initial_off",           est_sat_only_gen_rwmtq,       main_sat_rwmtq, 1,   w0,q0,h0,      cov0_estimate_only_gen_initial_off_rwmtq,      True,  quat_rwmtq_goals, prop_schedule,      4500,    orb_file,only_gen_int_cov_rwmtq,9,3,0,1e-3,0,"",0.0,1.0,0*w0,unitvecs4[0],0*h0]

case_quat_RWMTQ_all_eclipse = ["thesis_6U_quat_RWMTQ_est_w_all_eclipse",           est_sat_w_all_rwmtq_eclipse,     main_sat_rwmtq_eclipse,    1,  w0,      q0,h0,      cov0_estimate_all_rwmtq,      True,  quat_rwmtq_goals, prop_schedule,      4500,    orb_file,all_int_cov_rwmtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0,"",0.0,1.0,w0,q0,h0,True]
case_quat_RWMTQ_only_gen_eclipse = ["thesis_6U_quat_RWMTQ_est_w_only_gen_eclipse",           est_sat_only_gen_rwmtq_eclipse,       main_sat_rwmtq_eclipse, 1,  w0,      q0,h0,      cov0_estimate_only_gen_rwmtq,      True,  quat_rwmtq_goals, prop_schedule,      4500,    orb_file,only_gen_int_cov_rwmtq,9,3,0,1e-3,0,"",0.0,1.0,w0,q0,h0,True]


# tests = tests[6:]
tests = []
# tests += [case_quat_RWMTQ_all,case_quat_RWMTQ_all_nodipole,case_quat_RWMTQ_all_nobigrand,case_quat_RWMTQ_all_nodipole_nobigrand,case_quat_RWMTQ_all_al1kap0,case_quat_RWMTQ_all_al01kap0,case_quat_RWMTQ_all_al001kap0,case_quat_RWMTQ_all_al1kap3mL,case_quat_RWMTQ_all_1est0other,case_quat_RWMTQ_all_01est0other,case_quat_RWMTQ_all_001est0other,case_quat_RWMTQ_all_001est1other,case_quat_RWMTQ_all_1est1other]
# tests += [case_quat_RWMTQ_all_nodipole_0est_1other,case_quat_RWMTQ_all_nodipole_0est_01other,case_quat_RWMTQ_all_nodipole_0est_001other]
# tests += [case_quat_RWMTQ_all_nodipole_001est_001other_nobigrand,case_quat_RWMTQ_all_nodipole_01est_01other_nobigrand,case_quat_RWMTQ_all_nodipole_1est_01other_nobigrand,case_quat_RWMTQ_all_nodipole_1est_001other_nobigrand]
# tests += [case_quat_RWMTQ_all_nodipole_001est_001other,case_quat_RWMTQ_all_nodipole_01est_01other,case_quat_RWMTQ_all_nodipole_1est_01other,case_quat_RWMTQ_all_nodipole_1est_001other]
tests = [case_quat_RWMTQ_all,case_quat_RWMTQ_all_initial_off,case_quat_RWMTQ_only_gen,case_quat_RWMTQ_only_gen_initial_off,case_quat_RWMTQ_bias,case_quat_RWMTQ_bias_initial_off,case_quat_RWMTQ_dist,case_quat_RWMTQ_dist_initial_off,case_quat_RWMTQ_simple,case_quat_RWMTQ_simple_initial_off]
tests += [case_quat_RWMTQ_all_eclipse,case_quat_RWMTQ_only_gen_eclipse,case_quat_RWMTQ_bias_eclipse,case_quat_RWMTQ_dist_eclipse,case_quat_RWMTQ_simple_eclipse]



for j in tests:
    run_sim_wrapper(*j)
