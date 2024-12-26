#estimation results for paper
from sat_ADCS_estimation import *
from sat_ADCS_control import *
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from sat_ADCS_satellite import *
# from sat_ADCS_ADCS import *
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

from src.sat_ADCS_ADCS.ADCS import *
from src.sat_ADCS_ADCS.trajectory import *

with open("thesis_test_files/new_tests_marker"+time.strftime("%Y%m%d-%H%M%S")+".txt", 'w') as f:
    f.write("just a file to show when new runs of tests started.")

np.random.seed(1)
all_avcov = ((0.1*math.pi/180))**2.0
all_angcov = (0.05)**2.0
all_hcov = (1e-6)**2.0
all_avcov_initial_off = ((3.0*math.pi/180))**2.0
all_angcov_initial_off = 5.0**2.0#(100.0*math.pi/180)**2.0
all_hcov_initial_off = (1e-4)**2.0
all_werrcov = 1e-16#1e-20#1e-19
all_mrperrcov = 1e-10#1e-16#1e-16
all_herrcov = 1e-24#1e-24

# nothing_avcov = ((0.01*math.pi/180)/(60))**2.0
# nothing_angcov = (0.5*math.pi/180)**2.0
# nothing_hcov = (0.0001)**2.0
# nothing_werrcov = 1e-12
# nothing_mrperrcov = 1e-10
# nothing_herrcov = 1e-22

simple_avcov = ((0.1*math.pi/180)/(60))**2.0
simple_angcov = (0.1*math.pi/180)**2.0
simple_hcov = (1e-6)**2.0
simple_avcov_initial_off = ((1.0*math.pi/180))**2.0
simple_angcov_initial_off = (10.0)**2.0
simple_hcov_initial_off = (1e-4)**2.0
simple_werrcov = 1e-14
simple_mrperrcov = 1e-5
simple_herrcov = 1e-12


bias_avcov = ((0.1*math.pi/180))**2.0
bias_angcov = (0.1*math.pi/180)**2.0
bias_hcov = (1e-6)**2.0
bias_avcov_initial_off = ((5.0*math.pi/180))**2.0
bias_angcov_initial_off = (10.0)**2.0
bias_hcov_initial_off = (1e-4)**2.0
bias_werrcov = 1e-14
bias_mrperrcov = 1e-5
bias_herrcov = 1e-16


dist_avcov = ((0.1*math.pi/180))**2.0
dist_angcov = (0.1*math.pi/180)**2.0
dist_hcov = (1e-6)**2.0
dist_avcov_initial_off = ((5.0*math.pi/180))**2.0
dist_angcov_initial_off = (10.0)**2.0
dist_hcov_initial_off = (1e-4)**2.0
dist_werrcov = 1e-14
dist_mrperrcov = 1e-5
dist_herrcov = 1e-24

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
w0 = 0*2*np.array([0.01,0.01,0.001])#np.array([0.53,0.53,0.053])#/(180.0/math.pi)
h0 = -0.001*(15/1000)*np.ones(3)#np.array([-0.001*j.max_h for j in real_sat.actuators if isinstance(j,RW)])

mtm_scale = 1e0
mtq_bias_on = False
mtm_bias_on = False
gyro_bias_on = True
sun_bias_on = False
prop_torq_on = True
dipole_on = False
extra_random = True

prop_scale = 1e-6



#all sensor biases, dipole, prop torque

main_sat_rw =  create_GPS_6U_sat(real=True,rand=False,mtm_scale = mtm_scale,use_dipole = dipole_on,use_mtq = False,include_mtmbias = mtm_bias_on, include_gbias = gyro_bias_on,include_sbias = sun_bias_on,use_prop = prop_torq_on,prop_torq0 = 10*prop_scale*normalize(np.array([0.1,-8,1])))
est_sat_w_all_rw =  create_GPS_6U_sat(real=False,rand=False,use_prop = prop_torq_on, estimate_prop_torq = prop_torq_on,mtm_scale = mtm_scale,use_dipole = dipole_on, estimate_dipole = dipole_on, extra_big_est_randomness = extra_random ,use_mtq = False, include_mtqbias = mtq_bias_on,estimate_mtq_bias = mtq_bias_on,include_mtmbias = mtm_bias_on, include_gbias = gyro_bias_on,include_sbias = sun_bias_on)
est_sat_only_gen_rw =  create_GPS_6U_sat(real=False,rand=False,include_sbias = sun_bias_on,estimate_sun_bias = sun_bias_on,  include_mtmbias = mtm_bias_on,estimate_mtm_bias = mtm_bias_on, use_gg = False,use_drag = False, use_SRP = False,use_dipole=False,estimate_dipole = False,use_prop=False,estimate_prop_torq = False,use_gen=True,estimate_gen_torq=True,gen_torq_std =  (2-extra_random)*5e-6, gen_mag_max = 1e-3,mtm_scale = mtm_scale, extra_big_est_randomness = extra_random ,use_mtq = False, include_mtqbias = mtq_bias_on,estimate_mtq_bias = mtq_bias_on )

main_sat_rwmtq =  create_GPS_6U_sat(real=True,rand=False,mtm_scale = mtm_scale,use_dipole = dipole_on,use_mtq = True,include_mtmbias = mtm_bias_on, include_mtqbias = False,include_gbias = gyro_bias_on,include_sbias = sun_bias_on,use_prop = prop_torq_on,prop_torq0 = prop_scale*normalize(np.array([0.1,-8,1])))
est_sat_w_all_rwmtq =  create_GPS_6U_sat(real=False,rand=False,use_prop = prop_torq_on, estimate_prop_torq = prop_torq_on,mtm_scale = mtm_scale,use_dipole = dipole_on, estimate_dipole = dipole_on, extra_big_est_randomness = extra_random ,use_mtq = True, include_mtqbias = False,estimate_mtq_bias = False,include_mtmbias = mtm_bias_on, include_gbias = gyro_bias_on,include_sbias = sun_bias_on)
est_sat_only_gen_rwmtq =  create_GPS_6U_sat(real=False,rand=False,include_sbias = sun_bias_on,estimate_sun_bias = sun_bias_on,  include_mtmbias = mtm_bias_on,estimate_mtm_bias = mtm_bias_on, use_gg = False,use_drag = False, use_SRP = False,use_dipole=False,estimate_dipole = False,use_prop=False,estimate_prop_torq = False,use_gen=True,estimate_gen_torq=True,gen_torq_std =  (2-extra_random)*5e-6, gen_mag_max = 1e-3,mtm_scale = mtm_scale, extra_big_est_randomness = extra_random ,use_mtq = True, include_mtqbias = True,estimate_mtq_bias = True )


main_sat_mtq =  create_GPS_6U_sat(real=True,rand=False,mtm_scale = mtm_scale,use_dipole = dipole_on,use_mtq = True,include_mtmbias = mtm_bias_on, include_mtqbias = mtq_bias_on,include_gbias = gyro_bias_on,include_sbias = sun_bias_on,use_prop = prop_torq_on,use_RW = False,prop_torq0 = prop_scale*normalize(np.array([0.1,-8,1])))
est_sat_w_all_mtq =  create_GPS_6U_sat(real=False,rand=False,use_prop = prop_torq_on, estimate_prop_torq = prop_torq_on,mtm_scale = mtm_scale,use_dipole = dipole_on, estimate_dipole = dipole_on, extra_big_est_randomness = extra_random ,use_mtq = True, include_mtqbias = mtq_bias_on,estimate_mtq_bias = mtq_bias_on,include_mtmbias = mtm_bias_on, include_gbias = gyro_bias_on,include_sbias = sun_bias_on,use_RW = False)
est_sat_only_gen_mtq =  create_GPS_6U_sat(real=False,rand=False,include_sbias = sun_bias_on,estimate_sun_bias = sun_bias_on,  include_mtmbias = mtm_bias_on,estimate_mtm_bias = mtm_bias_on, use_gg = False,use_drag = False, use_SRP = False,use_dipole=False,estimate_dipole = False,use_prop=False,estimate_prop_torq = False,use_gen=True,estimate_gen_torq=True,gen_torq_std =  (2-extra_random)*5e-6, gen_mag_max = 1e-3,mtm_scale = mtm_scale, extra_big_est_randomness = extra_random ,use_mtq = True, include_mtqbias = mtq_bias_on,estimate_mtq_bias = mtq_bias_on ,use_RW = False)


main_sat_mtq1rw =  create_GPS_6U_sat_1RW(real=True,rand=False,mtm_scale = mtm_scale,use_dipole = dipole_on,use_mtq = True,include_mtmbias = mtm_bias_on, include_mtqbias = mtq_bias_on,include_gbias = gyro_bias_on,include_sbias = sun_bias_on,use_prop = prop_torq_on,use_RW = True,prop_torq0 = prop_scale*normalize(np.array([0.1,-8,1])))
est_sat_w_all_mtq1rw =  create_GPS_6U_sat_1RW(real=False,rand=False,use_prop = prop_torq_on, estimate_prop_torq = prop_torq_on,mtm_scale = mtm_scale,use_dipole = dipole_on, estimate_dipole = dipole_on, extra_big_est_randomness = extra_random ,use_mtq = True, include_mtqbias = mtq_bias_on,estimate_mtq_bias = mtq_bias_on,include_mtmbias = mtm_bias_on, include_gbias = gyro_bias_on,include_sbias = sun_bias_on,use_RW = True)
est_sat_only_gen_mtq1rw =  create_GPS_6U_sat_1RW(real=False,rand=False,include_sbias = sun_bias_on,estimate_sun_bias = sun_bias_on,  include_mtmbias = mtm_bias_on,estimate_mtm_bias = mtm_bias_on, use_gg = False,use_drag = False, use_SRP = False,use_dipole=False,estimate_dipole = False,use_prop=False,estimate_prop_torq = False,use_gen=True,estimate_gen_torq=True,gen_torq_std =  (2-extra_random)*5e-6, gen_mag_max = 1e-3,mtm_scale = mtm_scale, extra_big_est_randomness = extra_random ,use_mtq = True, include_mtqbias = mtq_bias_on,estimate_mtq_bias = mtq_bias_on ,use_RW = True)

orb_file = "lovera_orb_1"
axis_RW_goals = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.RWBDOT_WITH_EKF, 0.22+250*sec2cent:GovernorMode.RW_PID},
                {0.2:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3)), 0.22+1200*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3)), 0.22+2500*sec2cent:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3))},
                {0.22:-unitvecs[0], 0.22+1200*sec2cent:unitvecs[2], 0.22+2500*sec2cent:-unitvecs[0]})


plan_rwmtq_goals = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.RWBDOT_WITH_EKF, 0.22+150*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC},
                {0.2:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3)),
                0.22+1100*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),
                0.22+1200*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3)),
                0.22+1500*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),
                0.22+1600*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)),
                0.22+1900*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),
                0.22+2000*sec2cent:(PointingGoalVectorMode.POSITIVE_ORBIT_NORMAL,np.zeros(3)),
                0.22+2400*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),
                0.22+2500*sec2cent:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3))},
                {0.22:-unitvecs[0], 0.22+1200*sec2cent:unitvecs[2], 0.22+2500*sec2cent:-unitvecs[0]})

plan_rwmtq_goals_LQR = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.RWBDOT_WITH_EKF, 0.22+150*sec2cent:GovernorMode.PLAN_AND_TRACK_LQR},
                {0.2:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3)), 0.22+1100*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1200*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3)), 0.22+1500*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1600*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)), 0.22+1900*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2000*sec2cent:(PointingGoalVectorMode.POSITIVE_ORBIT_NORMAL,np.zeros(3)), 0.22+2400*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2500*sec2cent:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3))},
                {0.22:-unitvecs[0], 0.22+1200*sec2cent:unitvecs[2], 0.22+2500*sec2cent:-unitvecs[0]})


plan_mtq_goals = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF, 0.22+150*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC},
                {0.2:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3)),
                0.22+1100*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),
                0.22+1200*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3)),
                0.22+1500*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),
                0.22+1600*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)),
                0.22+1900*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),
                0.22+2000*sec2cent:(PointingGoalVectorMode.POSITIVE_ORBIT_NORMAL,np.zeros(3)),
                0.22+2400*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),
                0.22+2500*sec2cent:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3))},
                {0.22:-unitvecs[0],
                0.22+1200*sec2cent:unitvecs[2],
                0.22+2500*sec2cent:-unitvecs[0]})


plan_mtq_goals_LQR = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF, 0.22+150*sec2cent:GovernorMode.PLAN_AND_TRACK_LQR},
                {0.2:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3)), 0.22+1100*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1200*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3)), 0.22+1500*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1600*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)), 0.22+1900*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2000*sec2cent:(PointingGoalVectorMode.POSITIVE_ORBIT_NORMAL,np.zeros(3)), 0.22+2400*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2500*sec2cent:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3))},
                {0.22:-unitvecs[0], 0.22+1200*sec2cent:unitvecs[2], 0.22+2500*sec2cent:-unitvecs[0]})


plan_mtq_ECIgoals = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF, 0.22+150*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC},
                {0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR, -unitvecs[1]), 0.22+1100*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1200*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR, -unitvecs[0]), 0.22+1500*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1600*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[0]), 0.22+1900*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2000*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2]), 0.22+2400*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2500*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},
                {0.22:-unitvecs[0], 0.22+1200*sec2cent:unitvecs[2], 0.22+2500*sec2cent:-unitvecs[0]})


plan_mtq_ECIgoals_LQR = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF, 0.22+150*sec2cent:GovernorMode.PLAN_AND_TRACK_LQR},
                {0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,-unitvecs[1]), 0.22+1100*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1200*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR, -unitvecs[0]), 0.22+1500*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1600*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[0]), 0.22+1900*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2000*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2]), 0.22+2400*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2500*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},
                {0.22:-unitvecs[0], 0.22+1200*sec2cent:unitvecs[2], 0.22+2500*sec2cent:-unitvecs[0]})


plan_mtq1rw_goals = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF, 0.22+150*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC},
                {0.2:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3)), 0.22+1100*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1200*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3)), 0.22+1500*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1600*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)), 0.22+1900*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2000*sec2cent:(PointingGoalVectorMode.POSITIVE_ORBIT_NORMAL,np.zeros(3)), 0.22+2400*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2500*sec2cent:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3))},
                {0.22:-unitvecs[0], 0.22+1200*sec2cent:unitvecs[2], 0.22+2500*sec2cent:-unitvecs[0]})


MPC_test_goals = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF, 0.22+50*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC},
                    {0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR, -unitvecs[1]), 0.22+1100*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1200*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR, -unitvecs[0]), 0.22+1500*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+1600*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[0]), 0.22+1900*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2000*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2]), 0.22+2400*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)), 0.22+2500*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},
                    {0.22:-unitvecs[0], 0.22+1200*sec2cent:unitvecs[2], 0.22+2500*sec2cent:-unitvecs[0]})
#
# plan_mtq_goals_LQR = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF, 0.22+100*sec2cent:GovernorMode.PLAN_AND_TRACK_LQR},
#                 {0.2:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3))},
#                 {0.22:-unitvecs[0]})
#
#
# plan_mtq_goals = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF, 0.22+100*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC},
#                 {0.2:(PointingGoalVectorMode.ANTI_RAM,np.zeros(3))},
#                 {0.22:-unitvecs[0]})


#
# plan_mtq_goals = Goals({0.2:GovernorMode.NO_CONTROL, 0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF, 0.22+100*sec2cent:GovernorMode.PLAN_OPEN_LOOP},
#                 {0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[0])},
#                 {0.22:-unitvecs[0], 0.22+900*sec2cent:unitvecs[2], 0.22+1200*sec2cent:-unitvecs[0]})
#


                #first and third goal is pointing -x-axis anti-ram, +z-axis Nadir. second goal is +y-axis nadir, -x-axis anti-ram. based on wisniewski orbit convention and an imagined asteria

prop_schedule = {0.22+30*sec2cent:True, 0.22+50*sec2cent:False, 0.22+400*sec2cent:True, 0.22+1100*sec2cent:False, 0.22+2800*sec2cent:True}
# prop_schedule = {0.22+2*sec2cent:True, 0.22+15*sec2cent:False, 0.22+400*sec2cent:True}
# prop_schedule = {}

sens_cov0_scale = 1e-6
if mtm_bias_on:
    cov0_estimate_all_rw = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,all_hcov*np.eye(3),np.eye(3*mtm_bias_on)*(3e-6)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM) and j.has_bias]),np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(5e-3)**2.0)
    cov0_estimate_all_rwmtq = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,all_hcov*np.eye(3),np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*mtm_bias_on)*(3e-6)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM) and j.has_bias]),np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(5e-3)**2.0)
    cov0_estimate_all_mtq = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*mtm_bias_on)*(3e-6)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM) and j.has_bias]),np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(5e-3)**2.0)
    cov0_estimate_all_mtq_1rw = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,all_hcov*np.eye(1),np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*mtm_bias_on)*(3e-6)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM) and j.has_bias]),np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(5e-3)**2.0)

else:
    cov0_estimate_all_rw = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,all_hcov*np.eye(3),np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(5e-3)**2.0)

    cov0_estimate_all_rwmtq = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,all_hcov*np.eye(3),np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(5e-3)**2.0)
    cov0_estimate_all_mtq = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(5e-3)**2.0)
    cov0_estimate_all_mtq_1rw = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,all_hcov*np.eye(1),np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(gyro_bias_on*3)*(0.2*(math.pi/180.0))**2.0,np.eye(sun_bias_on*3)*(0.3*0.2)**2.0,0.15**2.0*np.eye(3*dipole_on),np.eye(prop_torq_on*3)*(5e-3)**2.0)



base_cov = block_diag(np.eye(3)*simple_avcov,np.eye(3)*simple_angcov,np.eye(3)*simple_hcov)
base_cov_norw = block_diag(np.eye(3)*simple_avcov,np.eye(3)*simple_angcov)
base_cov_1rw = block_diag(np.eye(3)*simple_avcov,np.eye(3)*simple_angcov,np.eye(1)*simple_hcov)
# cov0_estimate_gen = block_diag(np.eye(3)*all_avcov,np.eye(3)*all_angcov,np.eye(3)*all_hcov,np.eye(3)*(1e-7)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM)]),np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,0.15*np.eye(3),np.eye(3)*(1e-4)**2.0)
if mtm_bias_on:
    cov0_estimate_only_gen_rw = block_diag(base_cov,np.eye(3*mtm_bias_on)*(5e-7)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM)]),np.eye(3*gyro_bias_on)*(0.2*((math.pi/180.0)))**2.0,np.eye(3*sun_bias_on)*(0.3*0.2)**2.0,np.eye(3)*(5e-3)**2.0)

    cov0_estimate_only_gen_rwmtq = block_diag(base_cov,np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*mtm_bias_on)*(5e-7)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM)]),np.eye(3*gyro_bias_on)*(0.2*((math.pi/180.0)))**2.0,np.eye(3*sun_bias_on)*(0.3*0.2)**2.0,np.eye(3)*(5e-3)**2.0)
    cov0_estimate_only_gen_mtq = block_diag(base_cov_norw,np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*mtm_bias_on)*(5e-7)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM)]),np.eye(3*gyro_bias_on)*(0.2*((math.pi/180.0)))**2.0,np.eye(3*sun_bias_on)*(0.3*0.2)**2.0,np.eye(3)*(5e-3)**2.0)
    cov0_estimate_only_gen_mtq_1rw = block_diag(base_cov_1rw,np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*mtm_bias_on)*(5e-7)**2.0*np.diagflat([j.scale for j in main_sat_rwmtq.sensors if isinstance(j,MTM)]),np.eye(3*gyro_bias_on)*(0.2*((math.pi/180.0)))**2.0,np.eye(3*sun_bias_on)*(0.3*0.2)**2.0,np.eye(3)*(5e-3)**2.0)

else:
    cov0_estimate_only_gen_mtq = block_diag(base_cov_norw,np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*gyro_bias_on)*(0.2*((math.pi/180.0)))**2.0,np.eye(3*sun_bias_on)*(0.3*0.2)**2.0,np.eye(3)*(5e-3)**2.0)
    cov0_estimate_only_gen_mtq_1rw = block_diag(base_cov_1rw,np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*gyro_bias_on)*(0.2*((math.pi/180.0)))**2.0,np.eye(3*sun_bias_on)*(0.3*0.2)**2.0,np.eye(3)*(5e-3)**2.0)
    cov0_estimate_only_gen_rwmtq = block_diag(base_cov,np.eye(mtq_bias_on*3)*0.25**2.0,np.eye(3*gyro_bias_on)*(0.2*((math.pi/180.0)))**2.0,np.eye(3*sun_bias_on)*(0.3*0.2)**2.0,np.eye(3)*(5e-3)**2.0)
    cov0_estimate_only_gen_rw = block_diag(base_cov,np.eye(3*gyro_bias_on)*(0.2*((math.pi/180.0)))**2.0,np.eye(3*sun_bias_on)*(0.3*0.2)**2.0,np.eye(3)*(5e-3)**2.0)

dt = 1
mini_dist_ic = np.eye(0)# block_diag(*[j.std**2.0 for j in mini_est_sat_w_all.disturbances if j.estimated_param])
if prop_torq_on or dipole_on:
    all_dist_ic_rwmtq = block_diag(*[j.std**2.0 for j in est_sat_w_all_rwmtq.disturbances if j.estimated_param])
    all_dist_ic_rw = block_diag(*[j.std**2.0 for j in est_sat_w_all_rwmtq.disturbances if j.estimated_param])
else:
    all_dist_ic_rwmtq = np.eye(0)
    all_dist_ic_rw = np.eye(0)

all_dist_ic_rwmtq_nodipole = block_diag(*[j.std**2.0 for j in est_sat_w_all_rwmtq.disturbances if j.estimated_param])
# all_dist_ic_dipole = block_diag(*[j.std**2.0 for j in est_sat_w_all_w_dipole.disturbances if j.estimated_param])
only_gen_dist_ic_rwmtq = block_diag(*[j.std**2.0 for j in est_sat_only_gen_rwmtq.disturbances if j.estimated_param])



est_sat = est_sat_w_all_rwmtq
all_int_cov_rwmtq =  dt*block_diag(np.block([[np.eye(3)*all_werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*all_mrperrcov]]),dt*np.eye(3)*all_herrcov)#,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),all_dist_ic_rwmtq)
est_sat = est_sat_only_gen_rwmtq
only_gen_int_cov_rwmtq =  dt*block_diag(np.block([[np.eye(3)*only_gen_werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*only_gen_mrperrcov]]),dt*np.eye(3)*only_gen_herrcov)#,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),only_gen_dist_ic_rwmtq)


est_sat = est_sat_w_all_rw
all_int_cov_rw =  dt*block_diag(np.block([[np.eye(3)*all_werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*all_mrperrcov]]),dt*np.eye(3)*all_herrcov)#,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),all_dist_ic_rwmtq)
est_sat = est_sat_only_gen_rw
only_gen_int_cov_rw =  dt*block_diag(np.block([[np.eye(3)*only_gen_werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*only_gen_mrperrcov]]),dt*np.eye(3)*only_gen_herrcov)#,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),only_gen_dist_ic_rwmtq)


est_sat = est_sat_w_all_mtq
all_int_cov_mtq =  dt*block_diag(np.block([[np.eye(3)*all_werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*all_mrperrcov]]))#,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),all_dist_ic_rwmtq)
est_sat = est_sat_only_gen_mtq
only_gen_int_cov_mtq =  dt*block_diag(np.block([[np.eye(3)*only_gen_werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*only_gen_mrperrcov]]))#,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),only_gen_dist_ic_rwmtq)


est_sat = est_sat_w_all_mtq1rw
all_int_cov_mtq1rw =  dt*block_diag(np.block([[np.eye(3)*all_werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*all_mrperrcov]]),np.eye(1)*all_herrcov)#,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),all_dist_ic_rwmtq)
est_sat = est_sat_only_gen_mtq1rw
only_gen_int_cov_mtq1rw =  dt*block_diag(np.block([[np.eye(3)*only_gen_werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*only_gen_mrperrcov]]),np.eye(1)*only_gen_herrcov)#,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),only_gen_dist_ic_rwmtq)


case_quat_RWMTQ_vargoals = ["thesis_6U_quat_RWMTQ_vargoals",           est_sat_w_all_rwmtq,     main_sat_rwmtq,    1,  w0,      q0,h0,      cov0_estimate_all_rwmtq,      True,  plan_rwmtq_goals, prop_schedule,      1500,    orb_file,all_int_cov_rwmtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,w0,q0,h0]
case_quat_RWMTQ_vargoals_gen = ["thesis_6U_quat_RWMTQ_vargoals_wgen",           est_sat_only_gen_rwmtq,     main_sat_rwmtq,    1,  w0,      q0,h0,      cov0_estimate_only_gen_rwmtq,      True,  plan_rwmtq_goals, prop_schedule,      1500,    orb_file,all_int_cov_rwmtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,w0,q0,h0]
case_quat_RWMTQ_vargoals_LQR = ["thesis_6U_quat_RWMTQ_vargoals_LQR",           est_sat_w_all_rwmtq,     main_sat_rwmtq,    1,  w0,      q0,h0,      cov0_estimate_all_rwmtq,      True,  plan_rwmtq_goals_LQR, prop_schedule,      1500,    orb_file,all_int_cov_rwmtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,w0,q0,h0]

case_quat_RW_vargoals = ["thesis_6U_quat_RW_vargoals",           est_sat_w_all_rw,     main_sat_rw,    1,  w0,      q0,h0,      cov0_estimate_all_rw,      True,  plan_rwmtq_goals, prop_schedule,      3600,    orb_file,all_int_cov_rw,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),0,1e-3,0.0,"",0.0,1.0,w0,q0,h0]
case_quat_RW_vargoals_gen = ["thesis_6U_quat_RW_vargoals_wgen",           est_sat_only_gen_rw,     main_sat_rw,    1,  w0,      q0,h0,      cov0_estimate_only_gen_rw,      True,  plan_rwmtq_goals, prop_schedule,      3600,    orb_file,all_int_cov_rw,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on),0,1e-3,0.0,"",0.0,1.0,w0,q0,h0]
case_quat_RW_vargoals_LQR = ["thesis_6U_quat_RW_vargoals_LQR",           est_sat_w_all_rw,     main_sat_rw,    1,  w0,      q0,h0,      cov0_estimate_all_rw,      True,  plan_rwmtq_goals_LQR, prop_schedule,      3600,    orb_file,all_int_cov_rw,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),0,1e-3,0.0,"",0.0,1.0,w0,q0,h0]


case_quat_MTQ_vargoals_LQR = ["thesis_6U_quat_MTQ_vargoals_LQR",           est_sat_w_all_mtq,     main_sat_mtq,    1,  w0,      q0,[],      cov0_estimate_all_mtq,      True,  plan_mtq_goals_LQR, prop_schedule,      3600,    orb_file,all_int_cov_mtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,w0,q0,[]]
case_quat_MTQ_vargoals = ["thesis_6U_quat_MTQ_vargoals",           est_sat_w_all_mtq,     main_sat_mtq,    1,  w0,      q0,[],      cov0_estimate_all_mtq,      True,  plan_mtq_goals, prop_schedule,      3600,    orb_file,all_int_cov_mtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,w0,q0,[]]
case_quat_MTQ1RW_vargoals = ["thesis_6U_quat_MTQ1RW_vargoals",           est_sat_w_all_mtq1rw,     main_sat_mtq1rw,    1,  w0,      q0,h0[0],      cov0_estimate_all_mtq_1rw,      True,  plan_mtq_goals, prop_schedule,      3600,    orb_file,all_int_cov_mtq1rw,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,w0,q0,h0[0]]


case_quat_MTQ_varECIgoals_LQR = ["thesis_6U_quat_MTQ_varECIgoals_LQR",           est_sat_w_all_mtq,     main_sat_mtq,    1,  w0,      q0,[],      cov0_estimate_all_mtq,      True,  plan_mtq_ECIgoals_LQR, prop_schedule,      3600,    orb_file,all_int_cov_mtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,w0,q0,[]]
case_quat_MTQ_varECIgoals = ["thesis_6U_quat_MTQ_varECIgoals",           est_sat_w_all_mtq,     main_sat_mtq,    1,  w0,      q0,[],      cov0_estimate_all_mtq,      True,  plan_mtq_ECIgoals, prop_schedule,      3600,    orb_file,all_int_cov_mtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,w0,q0,[]]


MPC_debug_test_case  = ["MPC_debug",           est_sat_w_all_mtq,     main_sat_mtq,    1,  w0,      q0,[],      cov0_estimate_all_mtq,      True,  MPC_test_goals, prop_schedule,      500,    orb_file,all_int_cov_mtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,w0,q0,[]]

# case_BC_vargoals = ["thesis_BC_vargoals",     est_BC,     main_BC,    1,  w0,      q0,[],      cov0_estimate_BC,      True,  plan_mtq_goals, prop_schedule,      3600,    orb_file,all_int_cov_mtq,3*(mtm_bias_on+gyro_bias_on+sun_bias_on),3*(prop_torq_on+dipole_on),3*mtq_bias_on,1e-3,0.0,"",0.0,1.0,w0,q0,[]]


# tests = tests[6:]
tests = [case_quat_MTQ_vargoals,case_quat_RWMTQ_vargoals_LQR,case_quat_RWMTQ_vargoals,case_quat_MTQ1RW_vargoals,case_quat_MTQ_varECIgoals_LQR,case_quat_MTQ_varECIgoals,case_quat_MTQ_vargoals,case_quat_MTQ_vargoals_LQR,case_quat_MTQ1RW_vargoals,case_quat_RWMTQ_vargoals,case_quat_RWMTQ_vargoals_LQR,case_quat_RWMTQ_vargoals_gen,case_quat_RW_vargoals,case_quat_RW_vargoals_gen,case_quat_RW_vargoals_LQR]
tests = [case_quat_MTQ_vargoals]
for j in tests:
    run_sim_wrapper(*j)
