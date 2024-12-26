import numpy as np
import scipy
import random
import pytest
from sat_ADCS_estimation import *
from sat_ADCS_control import *
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from sat_ADCS_satellite import *
import warnings
# from flight_adcs.flight_utils.sat_helpers import *
# import sim_adcs.sim_helpers as sim_helpers#TODO: eliminate the need for this?
import math
import copy
import trajectory_planner.src.build.tplaunch as tplaunch
import trajectory_planner.src.build.pysat as pysat

from .trajectory import *

##todo: Test ADCS add/remove methods, control laws for planner, control modes

##check: bindings,  reduce some planner stuff.  input and output to C++,
##later:  Aero/SRP torque and derivatives to C++ dynamics.


"""
to update from ADCS folder

cd ../ADCS && \
python3.10 -m build && \
pip3.10 install ./dist/sat_ADCS_ADCS-0.0.1.tar.gz && \
cd ../ADCS
"""



class PlannerSettings:
    def __init__(self,sat,dt_tvlqr = 1,tvlqr_len = None,tvlqr_overlap = 1,dt_tp = None,precalculation_time = 100,default_traj_length = 1000,traj_overlap = 10,debug_plot_on = False,bdot_on = 1,
            include_gg = False,include_resdipole = False, include_prop = False, include_drag = False, include_srp = False, include_gendist = False):
        self.precalculation_time = precalculation_time
        #TODO: big one--this needs to be redone and cleaned up for different planners, etc.

        self.dt_tvlqr = dt_tvlqr
        if tvlqr_len is None:
            tvlqr_len = default_traj_length
        self.tvlqr_len = tvlqr_len
        if dt_tp is None:
            dt_tp = 10*dt_tvlqr
        self.tvlqr_overlap = tvlqr_overlap
        self.dt_tp = dt_tp
        self.default_traj_length = default_traj_length
        self.traj_overlap = traj_overlap
        self.debug_plot_on = debug_plot_on
        self.fig_exists = False
        self.bdot_on = bdot_on


        #TRAJECTORY PLANNER SETTINGS
        #Set qSettings
        self.bdotgain = 10000000#1000000000.
        self.gyrogainH = 0
        self.gyrogainL = 0
        #dampgain = -0.5#0.02#-0.5# -0.0002
        self.dampgainH = -2000.0#-0.5#0.02#-0.5# -0.0002
        self.dampgainL = -1000.0##0.02#-0.5# -0.0002
        self.velgainH =  -50.0#0.01#0.02#-0.1#-0.0001
        self.velgainL =  -200.0#0.01#-0.001#0.02#-0.1#-0.0001
        self.quatgainH =  -2.0#0.7#0*-1.0#1#0.1#0.1#0.0005#0.05#0.005#0.0005#0.01#0.00001
        self.quatgainL =  -0.001#-0.0005#0*-1.0#1#0.1#.001#0.0005#0.05#0.005#0.0005#0.01#0.00001x
        self.HLangleLimit = 10.0*math.pi/180
        self.Nslew = 0
        self.randvalH = 0.001#0.001#2.0#0.5
        self.randvalL = 0*0.00001
        self.umaxmultH = 1.5#1.5#3.0
        self.umaxmultL = 1.5

        '''THIS IS OLD, values are wrong, but the idea is correct-->due to units and constraints, the added cost per component per timestep can vary in magnitude.
        with a velocity magnitude limit of 0.006 rad/s (~0.34 deg/s), a time step with the satellite spinning at max rate will add ~2e-5 * swpoint to the cost due to velocity
        if the satellite is pointing 180 degrees in the wrong direction (based on vector pointing), the cost added in one timestep would be ~5e0 * sv1
        if the satellite is actuating all 3 MTQs at the limit of 0.15 Am^2, this will add ~3e-2 * su
        lagrange multipliers and penalties in their maximum (and with the added 100 multiplier on the omega^2 term) will each add costs per time step of roughly 1e9 to 1e12 if they are off by 1x their limit (u=2*umax, for example). Note that lagrange multiplier and penalty do not start at their maximum. The cost added per component per timestep does not need to compete with those, or be on the same scale. In fact, we should probably watch out for cost approaching the same scale, as that could mean that constraint enforcement may not overpower an infeasible but very optimal trajectory

        These differences should be taken into account when setting the values'''

        self.whichAngCostFunc = 0
        self.considerVectorInTVLQR = 0
        self.useRawControlCost = True
        #0 is (1-dot(ECIvec.T*rot_mat(q).T*satvec))
        #1 is 0.5*(1-dot(ECIvec.T*rot_mat(q).T*satvec))^2
        #2 is acos(dot(ECIvec.T*rot_mat(q).T*satvec)
        #3 is 0.5*acos(dot(ECIvec.T*rot_mat(q).T*satvec)^2

        self.quatvecmode = 0  #0 is 2*qv*sign(q0)/(1+|q0|), 1 is 2*qv/(1+q0), 2 is qv/q0

        self.mtq_control_weight = 0.0001
        self.rw_control_weight = 0.001
        self.magic_control_weight = 0.0001
        self.rw_AM_weight = 0.1
        self.rw_stic_weight = 0.01

        self.angle_weight = 10#0.1#200.0
        self.angvel_weight = 100#0*10.0#0.1#0.01#0*10.0#.01#01#0.000001
        self.u_weight_mult = 1.0#1e-1#1e-1 #0*0.0001#0.0#0.0000001
        self.u_with_mag_weight = 0.0
        self.av_with_mag_weight = 0#0.1
        self.ang_av_weight = 0*10.0#0*100.0
        self.angle_weight_N = 100#2000.0
        self.angvel_weight_N = 100#0*10.0#0.1#0*10.0
        self.av_with_mag_weight_N = 0.0
        self.ang_av_weight_N = 0*100.0

        self.angle_weight2 = 10#0.1#100.0
        self.angvel_weight2 = 0.1#0*1.0#0.0
        self.u_weight_mult2 = 1.0#1e-2#1# 1e-1
        self.u_with_mag_weight2 = 0.0
        self.av_with_mag_weight2 = 0.0
        self.ang_av_weight2 = 0*0.2#0*0.2#0.1#0.0001*0
        self.angle_weight_N2 = 1000#1000.0
        self.angvel_weight_N2 = 1.0#1.0#0*1.0#0.0
        self.av_with_mag_weight_N2 = 0
        self.ang_av_weight_N2 = 0*0.2#1.0#0*0.2

        self.angle_weight_tvlqr = 10#30#0.1#10.0
        self.angvel_weight_tvlqr = 10.0#0.1#0.01#0#0.001#0.01#0.01#0*0.01#1#$5#0.01#0.001#0.5#0.1#0#1#0*0.000001#1e-1#0*0.0000001#0.01#0.00001
        self.u_weight_mult_tvlqr = 1.0#1e-2#0.001#0.00001#0.001#0.00000001#0.3#5#0.1#0.000000001#0.0000000001
        self.u_with_mag_weight_tvlqr = 0.0
        self.av_with_mag_weight_tvlqr = 0#5.0#0#0.005#0.01
        self.ang_av_weight_tvlqr = 0.0#0*2#0.05#0.01#0.5#0.5#0#10#0.01#0.1#0*0.1#0*10#1000#-0.1#0.0001#0.01
        self.angle_weight_N_tvlqr = 100#1*500#1#500#200#30#20.0
        self.angvel_weight_N_tvlqr = 100#$1.0*10#0.1*10#0.1#0.1#0#0.01#1#1#0*0.1#0*0.01#1#5#0.01#0.001#0.5#0.1#0#1#0*0.000001#1e-1#0*0.0000001#0*0.00001
        self.av_with_mag_weight_N_tvlqr = 0
        self.ang_av_weight_N_tvlqr = 0.0#0*2#5.0#0#0.05*100#0*0.01#0.01#0.5#0.5#0#10#0.01#0.1#0.01#0.0

        self.sun_limit_angle = 20*3.14/180.0# 0.000000001#10*3.14/180.0 #RADIANS
        self.camera_axis = np.array([[1,0,0]]).T

        #Set forwardPassSettings
        self.maxLsIter = 20#25#30#30#20#20#45
        self.beta1 = 1e-10#1e-5#1e-2#1e-5#1e-10#1e-10#1e-10#1e-10#1e-10#1e-10#10#1e-10#1e-8#1e-2#1e-10#1e-4#1e-4#1e-4#2e-1
        self.beta2 = 20#10#5#2reg0 #20#50#20#100#50#50#25#25#15#25#.030.0 #5.0

        self.regScale = 1.6#1.6#1.6##2#1.6#4#1.6#2#1.6#1.8#2.0#1.8#5#10#1e3#1.8#2#1.6#1.6#2#1.6#2.0#1.6#3#1.6#5#2#5#1.6#5#1.8
        self.regMax = 1e10
        self.regMax2 = 1e12
        self.regMin = 1e-10#16
        self.regBump = 10#5#1e-3#1e-2#10.0#1e-1#2.0#10.0#1.0#10.0#1.0 #1.0#10.0#$0.01#0.1#2.0#1e-2#0.1#1.0#0.1#10#1e-10#2.0#1.0#1.0#1.0#10.0#100#0.1#10#1.0 #1.0#1.0#50#10#20.0#100.0
        self.regMinCond = 1 #0 means that regmin is basically ignored, and regulaization goes up and down without bounds, case 1 means regularization is always equal to or greater than regmin, case 2 means if the regularization falls below regmin then it clamps to 0
        self.regMinCond2 = 1
        self.regBumpRandAddRatio = 0#1e-20#1e-16#1e-3#4e-3#*1e-4
        self.useEVmagic = 0;#1 #use the eigendecomposition rather than simple regularization
        self.SPDEVreg = 1;#0#1 #regularize/add even if matrix is SPD
        self.SPDEVregAll = 0;#0 reg SPD matrix by adding rho*identity matrix (otherwise do the EV magic reg)
        self.rhoEVregTest = 1;#1 #test if reset is needed (in EV magic case) by comparing to a multiple of rho (otherwise compare to regmin)
        self.useDynamicsHess = 0
        self.EVregTestpreabs = 1;#0 #complete the reset test before absolute value is taken
        self.EVaddreg = 0;#0 #do EV magic by adding a value to the eigs that are too small (otherwise clamp to a minimum value)
        self.EVregIsRho = 1; #1 #clamp to or add rho (otherwise regmin)
        self.EVrhoAdd = 0;#0 #if adding a value to the eigs, this determines if the reg value added is added to the values less than rho (True) or regmin (false)
        self.useConstraintHess = 0 #

        self.control_limit_scale = 0.75
        self.umax = self.control_limit_scale*np.array([j.max for j in sat.actuators])#np.vstack([np.array(sat.MTQ_max).reshape((sat.number_MTQ,1)),np.array(sat.RW_torq_max).reshape((sat.number_RW,1))])
        #TODO: add RW saturation, other constraints.
        self.xmax = 10*np.ones((sat.state_len,1))
        self.eps = 2.22044604925031e-16
        self.satAlignVector = [0,0,1]
        self.wmax = 0.02#0.005 #rad/sec
        #Set backwardPassSettings#
        self.mu = 1.0
        self.rho = 0.0#1e-10#0.01#1.0#0.1#1.0#0.1#0.001#1.0#0.01#1.0
        self.drho = 1.0
        #Set alilqrSettings
        self.regInit = self.regMin#regMin#0#0.0#regMin#0.0#0.0#2*regMin

        self.maxOuterIter = 25#70#50#20#20#25#25#1#15#diop
        self.maxIlqrIter = 250#350#0#100#300#diop0#50#150#50#25#25#1#30
        self.maxOuterIter2 = 14#70#50#20#20#25#25#1#15#diop
        self.maxIlqrIter2 = 200#350#0#100#300#diop0#50#150#50#25#25#1#30

        self.maxIter = 4500
        self.maxIter2 = 3500
        self.gradTol = 1e-7#1e-9#1e-09
        self.costTol =      1e-9#1e-6#0.0000001
        self.ilqrCostTol =  1e-8#1e-4#0.000001
        self.maxCost = 1e10

        self.cmax = 0.002
        self.zCountLim = 20#14#20#10#30#10#45
        self.penInit = 1# 1.0#0.1#1.0#1.0#10#1.0#1.0#40.0#1.0#0.1#0.1#1#0.01#0.01#10.0#0.1#1#2.5#1e2#5e3#5.0#5#0.5#1.0F
        self.penInit2 = 1# 1.0#0.1#1.0#1.0#10#1.0#1.0#40.0#1.0#0.1#0.1#1#0.01#0.01#10.0#0.1#1#2.5#1e2#5e3#5.0#5#0.5#1.0F
        self.penMax = 1e10
        self.penScale = 10#5#10#10# 3#10#10#4#4#10#100#10#100.0#10.0

        self.lagMultInit = 0.0
        self.lagMultMax = 1e10
        self.lagMultMax2 = 1e10


        self.useACOSConstraint = 0
        self.useExtraAVConstraint = 0

        self.plan_for_aero = include_drag
        self.plan_for_prop = include_prop
        self.plan_for_srp = include_srp
        self.plan_for_gg = include_gg
        self.plan_for_gendist = include_gendist
        self.plan_for_resdipole = include_resdipole
        self.srp_coeff = np.zeros((3,))
        self.drag_coeff = np.zeros((3,))
        self.coeff_N = 0
        self.res_dipole =  sum([j.main_param if isinstance(j, Dipole_Disturbance) else np.zeros(3) for j in sat.disturbances ],start = np.zeros(3)).reshape((3,))
        self.prop_torque = sum([j.main_param if isinstance(j, Prop_Disturbance) else np.zeros(3) for j in sat.disturbances ],start = np.zeros(3)).reshape((3,))
        self.gendist_torq =  sum([j.main_param if isinstance(j, General_Disturbance) else np.zeros(3) for j in sat.disturbances ],start = np.zeros(3)).reshape((3,))
        self.J_est = sat.J

        self.RWh_max_mult = 0.8
        self.RWh_stiction_mult = 0.05
        self.RWh_ok_mult = 0.4
        self.verbosity = False

    def lineSearchSettings(self):
        return (self.maxLsIter,self.beta1,self.beta2)
    def auglagSettings(self):
        return (self.lagMultInit,self.lagMultMax,self.penInit,self.penMax,self.penScale)
    def breakSettings(self):
        return (self.maxOuterIter,self.maxIlqrIter,self.maxIter,self.gradTol,self.ilqrCostTol,self.costTol,self.zCountLim,self.cmax,self.maxCost,self.xmax)
    def regSettings(self):
        return (self.regInit,self.regMin,self.regMax,self.regScale,self.regBump,self.regMinCond,self.regBumpRandAddRatio,self.useEVmagic,self.SPDEVreg,self.SPDEVregAll,self.rhoEVregTest,self.EVregTestpreabs,self.EVaddreg,self.EVregIsRho,self.EVrhoAdd,self.useDynamicsHess,self.useConstraintHess)

    def lineSearchSettings2(self):
        return (self.maxLsIter,self.beta1,self.beta2)
    def auglagSettings2(self):
        return (self.lagMultInit,self.lagMultMax2,self.penInit2,self.penMax,self.penScale)
    def breakSettings2(self):
        return (self.maxOuterIter2,self.maxIlqrIter2,self.maxIter2,self.gradTol,self.ilqrCostTol,self.costTol,self.zCountLim,self.cmax,self.maxCost,self.xmax)
    def regSettings2(self):
        return (self.regInit,self.regMin,self.regMax2,self.regScale,self.regBump,self.regMinCond2,0*self.regBumpRandAddRatio,0,self.SPDEVreg,self.SPDEVregAll,self.rhoEVregTest,self.EVregTestpreabs,self.EVaddreg,self.EVregIsRho,self.EVrhoAdd,self.useDynamicsHess,self.useConstraintHess)

    def highSettings(self):
        return (self.gyrogainH,self.dampgainH,self.velgainH,self.quatgainH,self.randvalH,self.umaxmultH)
    def lowSettings(self):
        return (self.gyrogainL,self.dampgainL,self.velgainL,self.quatgainL,self.randvalL,self.umaxmultL)

    def systemSettings(self):
        return (self.J_est,self.dt_tp,self.dt_tvlqr,self.eps,self.tvlqr_len,self.tvlqr_overlap)
    def mainAlilqrSettings(self):
        return (self.lineSearchSettings(),self.auglagSettings(),self.breakSettings(),self.regSettings())
    def secondAlilqrSettings(self):
        return (self.lineSearchSettings2(),self.auglagSettings2(),self.breakSettings2(),self.regSettings2())
    def initTrajSettings(self):
        return (self.bdotgain,self.HLangleLimit,self.highSettings(),self.lowSettings())
    def optMainCostSettings(self):
        return (self.angle_weight,self.angvel_weight,self.u_weight_mult,self.u_with_mag_weight,self.av_with_mag_weight,self.ang_av_weight,self.angle_weight_N,self.angvel_weight_N,self.av_with_mag_weight_N,self.ang_av_weight_N,self.whichAngCostFunc,self.useRawControlCost)
    def optSecondCostSettings(self):
        return (self.angle_weight2,self.angvel_weight2,self.u_weight_mult2,self.u_with_mag_weight2,self.av_with_mag_weight2,self.ang_av_weight2,self.angle_weight_N2,self.angvel_weight_N2,self.av_with_mag_weight_N2,self.ang_av_weight_N2,self.whichAngCostFunc,self.useRawControlCost)
    def optTVLQRCostSettings(self,tracking_LQR_formulation):
        return (self.angle_weight_tvlqr,self.angvel_weight_tvlqr,self.u_weight_mult_tvlqr,self.u_with_mag_weight_tvlqr,self.av_with_mag_weight_tvlqr,self.ang_av_weight_tvlqr,self.angle_weight_N_tvlqr,self.angvel_weight_N_tvlqr,self.av_with_mag_weight_N_tvlqr,self.ang_av_weight_N_tvlqr,self.considerVectorInTVLQR,self.useRawControlCost,tracking_LQR_formulation)
    def planner_disturbance_settings(self):
        return ((self.plan_for_aero,self.plan_for_prop,self.plan_for_srp,self.plan_for_gg,self.plan_for_resdipole,self.plan_for_gendist),self.srp_coeff,self.drag_coeff,self.coeff_N,self.prop_torque,self.gendist_torq,self.res_dipole)


class Prop_Schedule:
    def __init__(self,prop_schedule_dict=None):
        # print(control_mode_dict)
        if prop_schedule_dict is None:
            prop_schedule_dict = {}
        self.prop_schedule = copy.deepcopy(dict(prop_schedule_dict))


@unique
class OUTSIDE_SOLVERS(Enum):
    DRAKE = 0
    CasADi = 1
    SciPy = 2



class ADCS():

    """
    ----=
    """
    def __init__(self,sat,orbit_estimator,attitude_estimator,control_laws,use_planner = False,planner = None,planner_settings = None,goals=None,prop_schedule=None,dt = 1,control_dt = None,tracking_LQR_formulation = 0,quatvecmode = 0,baseline_ctrl_mode = GovernorMode.BDOT_WITH_EKF,baseline_traj_ctrl_mode = GovernorMode.PLAN_AND_TRACK_LQR,outside_solver = OUTSIDE_SOLVERS.SciPy):
        """
        Initialize the governor.
        """
        self.virtual_sat = sat
        self.estimator = attitude_estimator
        self.orbit_estimator = orbit_estimator
        self.control_laws = control_laws
        self.quatvecmode = quatvecmode
        self.baseline_ctrl_mode = baseline_ctrl_mode
        self.baseline_traj_ctrl_mode = baseline_traj_ctrl_mode
        for j in self.control_laws:
            j.reset_sat(self.virtual_sat)

        if goals is None:
            self.goals = Goals()
        elif isinstance(goals,Goals):
            self.goals = goals
        elif isinstance(goals,list):
            self.goals = Goals(*goals)

        self.use_planner = use_planner

        if control_dt is None:
            self.control_dt = dt
        else:
            self.control_dt = control_dt
        self.tracking_LQR_formulation = tracking_LQR_formulation #0: 1, 0, 2

        if planner_settings is None:
            planner_settings = PlannerSettings(self.virtual_sat,self.control_dt)
        self.planner_settings = planner_settings
        if planner is None and self.use_planner:
            self.update_planner_settings()
            self.planner.setVerbosity(self.planner_settings.verbosity)
        if planner is not None:
            planner.setVerbosity(self.planner_settings.verbosity)
            self.planner = planner

        self.dt = dt
        self.addl_info = {}
        self.time_step =  self.gcd_sample_time([self.planner_settings.dt_tp, self.planner_settings.dt_tvlqr, self.estimator.update_period,self.control_dt,self.orbit_estimator.update_period,self.dt])

        self.nonmtq_ctrl_inds = np.nonzero(np.concatenate([np.ones(j.input_len)*(1-int(isinstance(j,MTQ))) for j in self.virtual_sat.actuators]))

        #during runtime
        self.current_trajectory = Trajectory()
        self.next_trajectory = Trajectory()
        self.precalculation_ready = False
        self.prop_status = 0
        self.prev_est_state = np.nan*np.ones(self.virtual_sat.state_len)
        # self.orbital_state = Orbital_State(0,np.array([0,0,1]),np.zeros(3))
        # self.prev_os = Orbital_State(0,np.array([0,0,1]),np.zeros(3))
        self.next_os = Orbital_State(0,np.array([0,0,1]),np.zeros(3))
        self.current_law = np.nan
        self.current_j2000 = np.nan
        self.j2000_delta = 0.25*sec2cent #amount of time that is counted as a different time and requires updating orbit guess
        self.prev_ctrl = np.nan*np.ones(self.virtual_sat.control_len)
        self.current_ctrl = np.zeros(self.virtual_sat.control_len)
        self.current_mode = ""
        self.current_goal = None
        self.next_goal = None
        self.prev_goal = None
        self.last_est_update_j2000 = 0
        self.last_ctrl_update_j2000 = 0
        self.last_informed_os_update_j2000 = 0
        self.planner_control_conversion = np.eye(self.virtual_sat.control_len)

        self.outside_solver = outside_solver


        self.prop_schedule = {}
        if prop_schedule is None:
            prop_schedule = {}
        self.add_prop_schedule(prop_schedule)

        self.len_lqr = np.nan
        if self.tracking_LQR_formulation==1:
            self.len_lqr = self.virtual_sat.state_len+1
        elif self.tracking_LQR_formulation==0:
            self.len_lqr = self.virtual_sat.state_len-1
        elif self.tracking_LQR_formulation==2:
            self.len_lqr = self.virtual_sat.state_len+2
        self.planned_info = [np.nan*np.ones(self.virtual_sat.state_len),np.nan*np.ones(self.virtual_sat.control_len),np.nan*np.ones((self.virtual_sat.control_len,self.len_lqr)),np.nan*np.ones((self.len_lqr,self.len_lqr)),np.nan*np.ones(3)]
        self.planned_next_info = [np.nan*np.ones(self.virtual_sat.state_len),np.nan*np.ones(self.virtual_sat.control_len),np.nan*np.ones((self.virtual_sat.control_len,self.len_lqr)),np.nan*np.ones((self.len_lqr,self.len_lqr)),np.nan*np.ones(3)]
        self.command = np.nan*np.ones(self.virtual_sat.control_len)
        self.prop_off(np.nan)
    #
    # def add_goals(self,new_goals): #TODO: not tested
    #     if new_goals is not None:
    #         if isinstance(new_goals,Goals):
    #             to_add = new_goals
    #         elif isinstance(new_goals,list):
    #             to_add = Goals(*new_goals)
    #     self.goals.add_goals(to_add)
    #
    # def replace_all_goals(self,new_goals):#TODO: not tested
    #     #TODO: how to deal with current goal, if conflicting
    #     if new_goals is None:
    #         self.goals = Goals()
    #     elif isinstance(new_goals,Goals):
    #         self.goals = new_goals
    #     elif isinstance(new_goals,list):
    #         self.goals = Goals(*new_goals)
    #
    # def clear_past_goals(self):#TODO: not tested
    #     self.goals.clear_past_goals(self.current_j200-self.j2000_delta)
    #
    # def replace_control_laws(self,new_laws):#TODO: not tested
    #     self.control_laws = new_laws
    #
    # def reset_sat(self,newsat):#TODO: not tested
    #     self.virtual_sat = sat
    #     for j in self.control_laws:
    #         j.reset_sat(self.virtual_sat)
    #     #TODO update estimator
    #
    # def replace_estimator(self,new_estimator):#TODO: not tested
    #     self.estimator = new_estimator
    #
    # def replace_orbit_estimator(self,new_orbit_estimator):#TODO: not tested
    #     self.orbit_estimator = new_orbit_estimator

    def add_prop_schedule(self,new_schedule):#TODO: not tested
        """
        This function adds new times to the prop schedule. It does NOT delete the existing ones or ones that have past.
        """
        cj2000 = self.current_j2000
        if np.isnan(cj2000):
            cj2000 = 0
        times = new_schedule.keys()
        if not len(times) == len(set(times)):
            raise ValueError("There is a repeated time in the prop schedule")
        if len(times)>0:
            if min(times)<cj2000:
                warnings.warn("schedule includes times that have already passed")

            noreps = {k:v for k,v in self.prop_schedule.items() if k not in new_schedule.keys()} #exclude repeats
            combined_dict = {**new_schedule,**noreps}
            times = sorted(combined_dict.keys())
            combined_dict = {k:combined_dict[k] for k in times} #make sure keys and values are sorted

            prop_switches = [2*(combined_dict[j]-0.5) for j in times if j>cj2000]
            prop_status = np.cumsum(np.array(prop_switches))
            prop_status = np.array(prop_status.flatten().tolist())
            if not (np.all(np.logical_or(prop_status==0,prop_status==1)) or (np.all(np.logical_or(prop_status==0,prop_status==-1)))):
                # breakpoint()
                raise ValueError("There are multiple instances of turning the prop on (or off) in a row.")
            if min(prop_status) == -1 and not self.prop_status:
                raise ValueError("The prop schedule and current prop status indicates that an already-off propulsion will be turned off.")
            if min(prop_status) == 0 and self.prop_status:
                raise ValueError("The prop schedule and current prop status indicates that an already-on propulsion will be turned on.")
            self.prop_schedule = combined_dict
        else:
            self.prop_schedule = {}


    def gcd_sample_time(self,block_sample_times, round_exp=4):
        """
        This function finds the desired common sample time by finding the GCD of a list of floats corresponding to sample
        times. Useful for finding a common sample time for TVLQR and the trajectory planner so we know what timestep to
        calculate the magnetic field at, etc.
        Inputs:
            block_sample_times -- list of floats
            round_exp -- number of places to round to
        Outputs:
            sim_sample_time -- max(1e-round_exp, appropriate sim sample time)
        """
        min_sample_time = 100000
        #Find number to round to
        round_num = 10**-round_exp
        #Find min GCD of 2 sample times
        return round(math.gcd(*[round(j/round_num) for j in block_sample_times])*round_num)
        # for i in range(len(block_sample_times)):
        #     for j in range(i, len(block_sample_times)):
        #         if np.gcd(block_sample_times[i], block_sample_times[j], round_num) < min_sample_time:
        #             min_sample_time = np.gcd(block_sample_times[i], block_sample_times[j], round_num)
        # #Return the rounded simulation sample time
        # return round(min_sample_time, round_exp)

    def replace_all_prop_schedule(self,new_schedule):#TODO: not tested
        """
        This function COMPLETELY replaces the prop schedule. It deletes all existing ones and adds the new ones.
        """
        times = new_schedule.keys()
        if not len(times) == len(set(times)):
            raise ValueError("There is a repeated time in the prop schedule")
        if min(times)<self.current_j2000:
            warnings.warn("schedule includes times that have already passed")

        times = sorted(new_schedule.keys())
        new_dict = {k:new_schedule[k] for k in times} #make sure keys and values are sorted

        prop_switches = [2*(new_dict[j]-0.5) for j in times if j>self.current_j2000]
        prop_status = np.cumsum(np.array(prop_switches))
        prop_status = prop_status.flatten().tolist()
        if not (np.all(np.logical_or(prop_status==0,prop_status==1)) or (np.all(np.logical_or(prop_status==0,prop_status==-1)))):
            raise ValueError("There are multiple instances of turning the prop on (or off) in a row.")
        if min(prop_status) == -1 and not self.prop_status:
            raise ValueError("The prop schedule and current prop status indicates that an already-off propulsion will be turned off.")
        if min(prop_status) == 0 and self.prop_status:
            raise ValueError("The prop schedule and current prop status indicates that an already-on propulsion will be turned on.")
        self.prop_schedule = new_dict

    def clear_past_prop_schedule(self):#TODO: not tested
        """
        This function clears out times in prop_schedule that have passed. (before current_j2000)
        """
        new = {k:self.prop_schedule[k] for k in sorted(self.prop_schedule.keys()) if k>=self.current_j2000} #make sure keys and values are sorted
        self.prop_schedule = new

    def ADCS_update(self,t,sens,GPS_sens,state_truth = None,orbital_truth = None,first_ADCS = False):#TODO: not tested
        self.addl_info.clear()
        self.current_j2000 = t
        sens = np.copy(sens)
        if state_truth is not None:
            state_truth = np.copy(state_truth)
        if orbital_truth is not None:
            orbital_truth = orbital_truth.copy()

        # print(t,self.orbit_estimator.prev_os.J2000,self.orbit_estimator.update_period)
        # print('os0',self.orbit_estimator.current_state_estimate.R)
        if (t-self.last_informed_os_update_j2000)*cent2sec>(self.orbit_estimator.update_period - time_eps):
            self.orbital_state_update(GPS_sens,t,orbital_truth)
        # print('os1',self.orbit_estimator.current_state_estimate.R)

        if (t-self.last_est_update_j2000)*cent2sec>(self.estimator.update_period - time_eps) and not first_ADCS:
            # print(self.orbit_estimation.R)
            self.estimation_update(sens,t,self.current_ctrl,truth = state_truth)



        plan_need,x_prec = self.check_for_precalculate_and_do()
        if plan_need:
            print('precalc done',x_prec)
            #do planning
            self.planner_settings.res_dipole = sum([j.main_param if isinstance(j, Dipole_Disturbance) else np.zeros(3) for j in self.virtual_sat.disturbances ],start = np.zeros(3))
            self.planner_settings.prop_torque = sum([j.main_param if isinstance(j, Prop_Disturbance) else np.zeros(3) for j in self.virtual_sat.disturbances ],start = np.zeros(3))
            self.planner_settings.gendist_torq = sum([j.main_param if isinstance(j, General_Disturbance) else np.zeros(3) for j in self.virtual_sat.disturbances ],start = np.zeros(3))
            self.update_planner_settings()
            lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,Tset_lqr = self.start_trajectory_planning(x_prec)


            # print(lqr_times-0.22)
            # print(self.tp_sat_vec)
            planned_orbit = self.tp_orbit.new_orbit_from_times(lqr_times)
            # pg,sv = self.goals.get_pointing_info(planned_orbit,quatmode = self.quatvecmode,for_TP = True)
            # sv = np.hstack([j for j in sv])
            # pg = np.hstack([j for j in pg])
            # # pointing = rot_mat_list(Xset_lqr[3:7,:],sv,transpose = False)

            Kmult = self.planner_control_conversion#np.eye(self.virtual_sat.control_len)
            Kmult[self.nonmtq_ctrl_inds,:] *= self.csat.read_magrw_torq_mult()
            Uset_lqr = self.planner_control_conversion@Uset_lqr
            Uset_lqr[self.nonmtq_ctrl_inds,:] *= self.csat.read_magrw_torq_mult()
            plot_the_thing(Uset_lqr[0:3,:],title = 'u cntrl',xdata = (lqr_times-0.22)*cent2sec,plot_now = True)
            plot_the_thing(Xset_lqr[0:3,:],title = 'av',xdata = (lqr_times-0.22)*cent2sec,plot_now = True)

            pt_err = np.arccos(np.array([np.dot(self.tp_sat_vec[j],rot_mat(Xset_lqr[3:7,j]).T@self.tp_point_goal[j]) for j in range(np.array(Xset_lqr).shape[1])]))*180.0/np.pi

            Sset_lqr = [Sset_lqr[:,j].reshape((self.len_lqr,self.len_lqr)) for j in range(Sset_lqr.shape[1])]
            Kset_lqr = [Kmult@Kset_lqr[:,j].reshape((self.virtual_sat.control_len,self.len_lqr)) for j in range(Kset_lqr.shape[1])]



            plot_the_thing(pt_err,title = 'ang err',xdata = (lqr_times-0.22)*cent2sec,plot_now = True)
            plot_the_thing(np.log10(pt_err),title = 'ang err',xdata = (lqr_times-0.22)*cent2sec,plot_now = True)
            #
            # KKeigs = np.stack([np.linalg.eigvals(j.T@j) for j in Kset_lqr])
            # CTGeigs = np.stack([np.linalg.eigvals(j) for j in Sset_lqr])
            # KKeigvecs0 = np.linalg.eig(Kset_lqr[0].T@Kset_lqr[0])[1]
            #
            # CTGeigvecs0 = np.linalg.eig(Sset_lqr[0])[1]
            # plot_the_thing(np.log10(np.abs(CTGeigs)),title = 'Log10 CTG eigs',xdata = (lqr_times-0.22)*cent2sec,plot_now = True)
            # plot_the_thing(CTGeigvecs0,title = 'CTG eigs',plot_now = True,scatter=True)
            #
            # plot_the_thing(np.log10(np.abs(KKeigs)),title = 'Log10 KTK eigs',xdata = (lqr_times-0.22)*cent2sec,plot_now = True)
            # plot_the_thing(KKeigvecs0.T,title = 'KTK eigs',plot_now = True,scatter=True)


            self.next_trajectory = Trajectory(lqr_times,Xset_lqr,Uset_lqr,Kset_lqr,Sset_lqr,Tset_lqr)
            self.precalculation_ready = True

        times = [t for t in self.prop_schedule.keys() if np.abs(self.current_j2000 - t)*cent2sec<time_eps]
        if len(times) > 1:
            raise ValueError("multiple prop commands")
        elif len(times) == 1:
            if bool(self.prop_schedule[t]):
                self.prop_on(t)
            else:
                self.prop_off(t)
        if (t-self.orbit_estimator.prev_os.J2000)*cent2sec>(self.dt - time_eps):
            self.command = self.actuation(t,sens)

        return self.command

    def update_planner_settings(self):
        csat = pysat.Satellite()
        csat.change_Jcom(self.virtual_sat.J)
        MTQs = [j for j in self.virtual_sat.actuators if isinstance(j,MTQ)]
        mtq_inds = [j for j in range(len(self.virtual_sat.actuators)) if isinstance(self.virtual_sat.actuators[j],MTQ)]
        RWs = [j for j in self.virtual_sat.actuators if isinstance(j,RW)]
        rw_inds = [j for j in range(len(self.virtual_sat.actuators)) if isinstance(self.virtual_sat.actuators[j],RW)]
        Magics = [j for j in self.virtual_sat.actuators if isinstance(j,Magic)]
        magic_inds = [j for j in range(len(self.virtual_sat.actuators)) if isinstance(self.virtual_sat.actuators[j],Magic)]
        ctrl_mode = self.control_laws[0]
        cc = 0*np.eye(self.virtual_sat.control_len)
        for k in MTQs:
            csat.add_MTQ(k.axis,self.planner_settings.control_limit_scale*k.max,self.planner_settings.mtq_control_weight)
        for k in RWs:
            csat.add_RW(k.axis,k.J,self.planner_settings.control_limit_scale*k.max,self.planner_settings.RWh_max_mult*k.max_h,self.planner_settings.rw_control_weight,self.planner_settings.rw_AM_weight,self.planner_settings.RWh_stiction_mult*k.max_h,self.planner_settings.rw_stic_weight,self.planner_settings.RWh_ok_mult*k.max_h)
        for k in Magics:
            csat.add_magic(k.axis,self.planner_settings.control_limit_scale*k.max,self.planner_settings.magic_control_weight)
        for k in range(len(mtq_inds)):
            cc[mtq_inds[k],k] = 1.0
        for k in range(len(rw_inds)):
            cc[rw_inds[k],k+len(mtq_inds)] = 1.0
        for k in range(len(magic_inds)):
            cc[magic_inds[k],k+len(mtq_inds)+len(rw_inds)] = 1.0
        self.planner_control_conversion = cc
        if self.planner_settings.wmax>0:
            csat.set_AV_constraint(self.planner_settings.wmax)
        if self.planner_settings.sun_limit_angle>0:
            csat.add_sunpoint_constraint(self.planner_settings.camera_axis,self.planner_settings.sun_limit_angle,0)
        if self.planner_settings.plan_for_gg:
            csat.add_gg_torq()
        if self.planner_settings.plan_for_aero:
            csat.add_aero_torq(self.planner_settings.drag_coeff,self.planner_settings.coeff_N)
        if self.planner_settings.plan_for_srp:
            csat.add_srp_torq(self.planner_settings.srp_coeff,self.planner_settings.coeff_N)
        if self.planner_settings.plan_for_resdipole:
            csat.add_resdipole_torq(self.planner_settings.res_dipole.reshape((3,1)))
        if self.planner_settings.plan_for_prop:
            csat.add_prop_torq(self.planner_settings.prop_torque.reshape((3,1)))
        if self.planner_settings.plan_for_gendist:
            csat.add_gendist_torq(self.planner_settings.gendist_torq.reshape((3,1)))

        planner = tplaunch.Planner(csat,self.planner_settings.systemSettings(),
                                    self.planner_settings.mainAlilqrSettings(),
                                    self.planner_settings.secondAlilqrSettings(),
                                    self.planner_settings.initTrajSettings(),
                                    self.planner_settings.optMainCostSettings(),
                                    self.planner_settings.optSecondCostSettings(),
                                    self.planner_settings.optTVLQRCostSettings(self.tracking_LQR_formulation))
        planner.setquaternionTo3VecMode(self.planner_settings.quatvecmode)
        self.planner = planner
        self.csat = csat
        self.planner.setVerbosity(self.planner_settings.verbosity)

    def estimation_update(self,sens,j2000,control,truth = None):#TODO: not tested
        # print('est0', self.orbit_estimator.prev_os.J2000,self.orbit_estimator.current_state_estimate.J2000,self.next_os.J2000)
        # print(self.orbit_estimator.prev_os.R,self.orbit_estimator.current_state_estimate.R,self.next_os.R)
        if np.abs(self.current_j2000 - j2000) > self.j2000_delta:
            warnings.warn('orbital state estimate is out of date--updating via propagation')
            self.orbit_estimator.propagate(j2000)

            self.orbital_state = self.orbit_estimator.current_state_estimate
        if control is None:
            control = self.current_ctrl

        # print('est1', self.orbit_estimator.prev_os.J2000,self.orbit_estimator.current_state_estimate.J2000,self.next_os.J2000)
        # print(self.orbit_estimator.prev_os.R,self.orbit_estimator.current_state_estimate.R,self.next_os.R)
        self.prev_est_state = np.copy(self.estimator.use_state.val)
        self.estimator.update(control,sens,self.orbit_estimator.current_state_estimate,truth = truth)
        time_step = (j2000-self.last_est_update_j2000)*cent2sec
        self.last_est_update_j2000 = j2000
        # print('est2', self.orbit_estimator.prev_os.J2000,self.orbit_estimator.current_state_estimate.J2000,self.next_os.J2000)
        # print(self.orbit_estimator.prev_os.R,self.orbit_estimator.current_state_estimate.R,self.next_os.R)

    # def orbital_state_propagate(self,j2000):#TODO: not tested
    #     current_os = self.orbital_state.orbit_rk4((j2000-self.orbital_state.J2000)*cent2sec)
    #     # self.prev_os = self.orbital_state.copy()
    #
    #     self.orbital_state = current_os
    #     self.current_j2000 = self.orbital_state.J2000
    #     if dt is None:
    #         dt = self.dt
    #     self.next_os = current_os.orbit_rk4(self.control_dt,calc_B = False,calc_S = False,calc_all = False)

    def orbital_state_update(self,GPS_sens,j2000,dt = None,truth = None):#TODO: not tested
        if truth is not None:
            truth = truth.copy()
        current_os = self.orbit_estimator.update(GPS_sens,j2000,truth)

        # self.orbital_state = current_os
        self.current_j2000 = self.orbit_estimator.current_state_estimate.J2000
        if dt is None:
            dt = self.dt
        self.next_os = current_os.orbit_rk4(self.control_dt,calc_B = True,calc_S = True,calc_all = True)
        self.last_informed_os_update_j2000 = self.current_j2000
        self.orbital_state = self.orbit_estimator.current_state_estimate

    def actuation(self,j2000,sens,fake = False,save_ctrl = True):#TODO: not tested
        if np.abs(self.current_j2000 - j2000) > self.j2000_delta:
            warnings.warn('orbital state estimate is out of date--updating via propagation')
            self.orbit_estimator.propagate(j2000)
            self.orbital_state = self.orbit_estimator.current_state_estimate
        self.current_mode = self.goals.get_control_mode(self.current_j2000)
        self.current_law = [j for j in self.control_laws if j.modename==self.current_mode][0]
        self.prev_goal = self.current_goal
        self.current_goal = self.goals.get_pointing_info(self.orbit_estimator.current_state_estimate)
        self.next_goal = self.goals.get_pointing_info(self.next_os)


        if self.current_mode in PlannerModeList:
            if self.precalculation_ready:
                print('testing switch')
                if self.next_trajectory.time_in_span(j2000): #in correct time block
                    self.switch_to_precalculated_trajectory(True, self.estimator.use_state.val) #True for verbose

            [ps,pc,pg,pm,pt] = self.current_trajectory.info_nearest_to_time(self.current_j2000)
            self.planned_info = [ps,pc,pg,pm,pt]
            cg = state_goal(goal_q=ps[3:7],goal_w=ps[0:3],goal_extra=ps[7:self.virtual_sat.state_len],u=self.current_goal.eci_vec,v=self.current_goal.body_vec)
            ctrl_goal = (ps,pc,pg,pm,pt,self.current_goal.eci_vec,self.current_goal.body_vec,cg)
            self.current_goal = cg


            [psp1,pcp1,pgp1,pmp1,ptp1] = self.current_trajectory.info_nearest_to_time(self.next_os.J2000)
            ng =  state_goal(goal_q=psp1[3:7],goal_w=psp1[0:3],goal_extra=psp1[7:self.virtual_sat.state_len],u=self.next_goal.eci_vec,v=self.next_goal.body_vec)
            self.planned_next_info = [psp1,pcp1,pgp1,pmp1,ptp1]
            ctrl_goalp1 = (psp1,pcp1,pgp1,pmp1,ptp1,self.next_goal.eci_vec, self.next_goal.body_vec,ng)
            self.next_goal = ng
        else:
            if not self.current_trajectory.is_empty():
                if self.current_trajectory.max_time() < (j2000 - self.dt*sec2cent): #get rid of old trajectory now that we are past it.
                    self.switch_to_precalculated_trajectory(False)
            # goalp1, svp1 = self.goals.get_pointing_info(osp1,quatmode = self.controller.quatvecmode)
            if self.current_mode in fullQuatSpecifiedList:
                ctrl_goal = ([],[],[],[],[],self.current_goal.eci_vec,self.current_goal.body_vec,self.current_goal)
                ctrl_goalp1 = ([],[],[],[],[],self.next_goal.eci_vec, self.next_goal.body_vec,self.next_goal)
                # if self.current_mode in QuaternionModeList:
                #     # self.planned_state = self.current_goal.state#np.vstack([np.zeros(3),vec3_to_quat(ctrl_goal[5],self.quatvecmode)])
                #     self.planned_next_state = self.next_goal.state#np.vstack([np.zeros(3),vec3_to_quat(ctrl_goalp1[5],self.quatvecmode)])
            else:
                ctrl_goal = ([],[],[],[],[],self.current_goal.eci_vec,self.current_goal.body_vec,self.current_goal)
                ctrl_goalp1 = ([],[],[],[],[],self.next_goal.eci_vec, self.next_goal.body_vec,self.next_goal)
                # if self.current_mode in QuaternionModeList:
                #     # self.planned_state = np.concatenate([np.zeros(3),vec3_to_quat(ctrl_goal[5],self.quatvecmode)])
                #     self.planned_next_state = np.concatenate([np.zeros(3),vec3_to_quat(ctrl_goalp1[5],self.quatvecmode)])


        control = self.current_law.find_actuation(self.estimator.use_state.val,self.orbit_estimator.current_state_estimate,self.next_os,self.current_goal,self.prev_goal,self.next_goal,sens,(ctrl_goal,ctrl_goalp1),fake)

        if save_ctrl and not fake:
            self.prev_ctrl,self.current_ctrl = self.current_ctrl,control
        if not fake:
            self.last_ctrl_update_j2000 = j2000
        return control

    def prop_off(self,j2000): #
        """
        This function turns propulsion off for the TP, attitude EKF, and controller.
        """
        self.prop_status = False
        self.virtual_sat.prop_dist_off() #TODO: multiple prop sources
        self.estimator.prop_off(j2000) #TODO: this estimator function may be redundant with the virtual sat.


    def prop_on(self,j2000): #
        """
        This function turns propulsion on for the TP, virutal sat, etc.

        Note that if plan_prop_on is still False, the TP won't plan with prop. This just *enables*
        modules to use propulsion, but doesn't force them to.
        """
        self.prop_status = True
        self.virtual_sat.prop_dist_on() #TODO: multiple prop sources
        self.estimator.prop_on(j2000) #TODO: this estimator function may be redundant with the virtual sat.


    def propulsion_finder(self,orbit,prev_t,prev_p):
        #"prev" meaning right before start of this orbital segment in "orbit".
        #TODO: not tested.
        prop_prec_t = np.array(sorted(orbit.times)) #all times in the orbit
        prop_prec_p = np.array([float(prev_p)]*len(prop_prec_t))  #all propulsion states in the orbit, set as equal to the state going in
        prop_times = sorted([t for t in self.prop_schedule.keys() if prev_t<t and t<=max(prop_prec_t)]) #check propulsion schedule for any switches that occur between now and the course of the orbit--including time before the orbit starts for precalculation.
        prop_switches = [bool(self.prop_schedule[t]) for t in prop_times] #find the values that the propulsion switches to during those times
        #The propulsion schedule should be turning on when off and off when on. This next bit checks for this by checking every other command is off or on
        #TODO deal with error states on this or allow for it to be turned off when already off (if it went into safe mode for example)
        evens = prop_switches[0::2] #
        odds = prop_switches[1::2]
        if prev_p: #prop already on
            if any(evens): #all evens should be "False", turning prop off
                raise ValueError("seems prop is being turned on when already on")
            if not all(odds): #all odds should be "True", turning prop on
                raise ValueError("seems prop is being turned off when already off")
        if not prev_p: #prop already on
            if not all(evens): #all evens should be "True", turning prop on
                raise ValueError("seems prop off being turned off when already off")
            if any(odds): #all odds should be "False", turning prop off
                raise ValueError("seems prop is being turned on when already on")

        for j in prop_times:
            #for each othe switching times in prop_times, set all prop states for subsequent times to the swtiched value--overwrite later ones repeatedly for simplicity instead of find segments between switch times.
            prop_prec_p[prop_prec_t>j] = int(self.prop_schedule[j])
        return prop_prec_t,prop_prec_p

    def precalculate_for_trajectory_planner(self, t_start_planning, t_end_planning):
        """
        This function precalculates r_ECI, v_ECI, B_ECI, pointing_goal, and x_est over the timespan
        t_start_planning to t_end_planning, based on the current state estimate and pointing goal mode.

        Parameters
        ---------------
            x_est: 7 x 1 np array, represents current estimated state
            rv_est: 6 x 1 np array, represents current estimate of r_ECI and v_ECI from orbit Ekf
            t_current: tuple, of format (decimal_hours, day, month, year), current time at start of precalculation
            t_start_planning: tuple, of format (decimal_hours, day, month, year), traj start time
            t_end_planning: tuple, of format (decimal_hours, day, month, year), traj end time

        Returns
        --------------
            point_ECI: 8 x N np array, array of ECI pointing goal vectors to feed into TP (w/ times)
            sat_vec: 8 x N np array, array of satellite body pointing vecs to feed into TP (w/ times)
            B_ECI: 8 x N np array, array of magfield vecs to feed into TP (w/ times)
            r_ECI: 8 x N np array, array of orbital position vecs to feed into TP (w/ times)
            v_ECI: 8 x N np array, array of orbital velocity vecs to feed into TP (w/ times)
            sun_ECI: 8 x N np array, array of sun vecs rel. to s/c pos to feed into TP (w/ times)
            x_trajstart: 7 x 1 np array, [w, q]^T est. state at start time of trajectory
        """
        print('starting to precaculate')
        #We need to find an appropriate dt for r_ECI, v_ECI, which is the largest common sample time of
        #the TP sample time, TVLQR sample time, and controller sample time (for precalculation)
        #TODO: give precalculation its own sample time?
        gcd_dt_precalc = self.gcd_sample_time([self.planner_settings.dt_tp, self.planner_settings.dt_tvlqr, self.estimator.update_period,self.dt,self.orbit_estimator.update_period])
        max_dt = max([self.planner_settings.dt_tp, self.planner_settings.dt_tvlqr, self.estimator.update_period,self.dt,self.orbit_estimator.update_period])
        #j2000_current = J2000calc(t_current[0], t_current[1], t_current[2], t_current[3])
        #j2000_start_planning = J2000calc(t_start_planning[0], t_start_planning[1], t_start_planning[2], t_start_planning[3])
        #j2000_end_planning = J2000calc(t_end_planning[0], t_end_planning[1], t_end_planning[2], t_end_planning[3])

        j2000_start_precalc = self.current_j2000#j2000_update(j2000_current, -1*max_dt)
        j2000_end_precalc = t_end_planning + 1.0*max_dt/cent2sec
        # print(t_end_planning)

        # print((t_end_planning-self.current_j2000)*cent2sec,gcd_dt_precalc)
        orbit_prec = Orbit(self.orbit_estimator.current_state_estimate,end_time = j2000_end_precalc,dt = gcd_dt_precalc)
        self.orbit_projected = orbit_prec

        # if j2000_start_precalc > 0.22+100*sec2cent:
        #     ost = self.tp_orbit.get_os(t_start_planning)
        #     goalt = self.goals.get_pointing_info(ost,for_TP=False,quatmode = self.quatvecmode)
        #     point_ECI = goalt.eci_vec
        #     sat_vec = goalt.body_vec
        #     print("******Expected error****")
        #     print((t_start_planning-0.22)*cent2sec)
        #     print((ost.J2000-0.22)*cent2sec)
        #     print(point_ECI.T)
        #     print(ost.R,ost.V)
        #     print(sat_vec.T)
        #     x_prec_tmp = self.current_trajectory.state_nearest_to_time(t_start_planning)
        #     print(x_prec_tmp[3:7].T)
        #     print(np.arccos(np.dot(sat_vec,rot_mat(x_prec_tmp[3:7]).T@point_ECI))*180.0/np.pi)

        gcd_dt_tp = self.gcd_sample_time([self.planner_settings.dt_tp, self.planner_settings.dt_tvlqr])
        tp_orbit = orbit_prec.get_range(t_start_planning,t_end_planning,dt = gcd_dt_tp)
        point_ECI,sat_vec = self.goals.get_pointing_info(tp_orbit,for_TP=True,quatmode = self.quatvecmode)
        self.tp_orbit = tp_orbit
        self.tp_point_goal = point_ECI
        self.tp_sat_vec = sat_vec

        # if j2000_start_precalc > 0.22+50*sec2cent:
        #     print("tp0",self.tp_point_goal[0].T)
        #     print((tp_orbit.min_time()-0.22)*cent2sec)
        #     osb = tp_orbit.get_os(tp_orbit.min_time())
        #     print(self.goals.get_pointing_info(osb,for_TP=True,quatmode = self.quatvecmode).eci_vec.T)
        #     print(self.goals.get_pointing_info(osb,for_TP=True,quatmode = self.quatvecmode).body_vec.T)
        #     print(osb.R,osb.V)
        #     print(np.arccos(np.dot(self.goals.get_pointing_info(osb,for_TP=True,quatmode = self.quatvecmode).body_vec,rot_mat(x_prec_tmp[3:7]).T@self.goals.get_pointing_info(osb,for_TP=True,quatmode = self.quatvecmode).eci_vec))*180.0/np.pi)
        # #     breakpoint()

        prev_t = self.orbit_estimator.prev_os.J2000 #assumes no prop switch in the time between the previous estimation and now.
        prop_prec_t,prop_prec_p = self.propulsion_finder(orbit_prec,prev_t,self.prop_status)
        ind_before_tp = np.where(np.array(prop_prec_t)<=t_start_planning)[0][-1]
        t_before_tp = prop_prec_t[ind_before_tp]
        p_before_tp = prop_prec_p[ind_before_tp]
        prop_tp_t,prop_tp_p = self.propulsion_finder(tp_orbit,t_before_tp,p_before_tp)

        self.tp_prop = prop_tp_p

        # print('orbit generated')
        # self.prop_projected = prop_prec

        #Next, calculate x_trajstart
        #t = t_current
        x_prec = self.estimator.use_state.val
        x_prec_prev = self.prev_est_state
        estimated = False
        self.prec_t = []
        self.prec_X = []
        self.prec_U = []
        #Update x_prec if we are currently running a trajectory
        if not self.current_trajectory.is_empty():
            t = j2000_start_precalc
            actual_prop_status = self.prop_status
            while t<t_start_planning:
                ctrl_mode = self.goals.get_control_mode(t)
                #If in plan and track but traj doesn't exist, do bdot
                if ctrl_mode in PlannerModeList:
                    ctrl_mode = GovernorMode.PLAN_AND_TRACK_LQR
                #Get B_ECI, r_ECI at t, t-1, and t+1
                os = orbit_prec.get_os(t)
                osp1 = orbit_prec.get_os(t+self.time_step/cent2sec)
                # osm1 = orbit_prec.get_os(t-self.time_step/cent2sec)
                #breakpoint()
                    #dB_body = dB_body#np.atleast_2d(np.array(np.matrix(((rot.T@Bt_ECI -rot_mat(x_prec_prev[-4:]).T@Bt_ECI)/self.sample_time)).A1)).reshape(3)
                #Get control vector by simulating whatever mode we're in (not PLAN_AND_TRACK).
                [ps,pc,pg,pm,pt] = self.current_trajectory.info_nearest_to_time(self.current_j2000)
                cg = state_goal(goal_q=ps[3:7],goal_w=ps[0:3],goal_extra=ps[7:self.virtual_sat.state_len],u=self.current_goal.eci_vec,v=self.current_goal.body_vec)
                ctrl_goal = (ps,pc,pg,pm,pt,self.current_goal.eci_vec,self.current_goal.body_vec,cg)


                [psp1,pcp1,pgp1,pmp1,ptp1] = self.current_trajectory.info_nearest_to_time(self.next_os.J2000)
                ng =  state_goal(goal_q=psp1[3:7],goal_w=psp1[0:3],goal_extra=psp1[7:self.virtual_sat.state_len],u=self.next_goal.eci_vec,v=self.next_goal.body_vec)
                ctrl_goalp1 = (psp1,pcp1,pgp1,pmp1,ptp1,self.next_goal.eci_vec, self.next_goal.body_vec,ng)

                # prop_inds = np.where([j<t for j in prop_prec_t])[0]
                # if len(prop_inds) == 0:
                #     raise ValueError("There should always be a time in the list less than this...")
                # prop_t = prop_prec_p[max(prop_inds)]
                prec_law = [j for j in self.control_laws if j.modename==ctrl_mode][0]
                u_prec = prec_law.find_actuation(self.estimator.use_state.val,self.orbit_estimator.current_state_estimate,self.next_os,self.current_goal,self.prev_goal,self.next_goal,np.nan,(ctrl_goal,ctrl_goalp1),True)
                #Now, propagate attitude forward using rk4 + normalize quaternion
                #Increment t
                dt = min(abs(t_start_planning-t)*cent2sec,self.time_step)
                x_prec_prev = x_prec
                prop_j = prop_prec_p[np.where(np.array(prop_prec_t)<=t)[0][-1]]
                use_prop_j = prop_j and self.planner_settings.plan_for_prop
                self.prec_t += [t]
                self.prec_X += [x_prec]
                self.prec_U += [u_prec]
                if use_prop_j:
                    self.virtual_sat.prop_dist_on()
                else:
                    self.virtual_sat.prop_dist_off()
                x_prec = self.virtual_sat.rk4(x_prec[0:self.virtual_sat.state_len], u_prec, gcd_dt_precalc, os,osp1)
                #TODO: add disturbance estimation here?
                t += dt/cent2sec
            estimated = True


            # if self.current_trajectory.time_in_span(t_start_planning):
            #     #t_start_planning is in current trajectory
            #     x_prec = self.current_trajectory.state_nearest_to_time(t_start_planning)
            #     print('using best point in current trajectory')
            #     estimated = True
            # elif (self.precalculation_ready and self.next_trajectory.time_in_span(t_start_planning)):
            #     #t_start_planning is in the next planned trajectory??? Not sure this state can arise
            #     x_prec = self.next_trajectory.state_nearest_to_time(t_start_planning)
            #     estimated = True

        if not estimated:
            #TODO: could error if it searches for a future trajectory that doesn't exist?? This could happen if precalc was longer than trajectories
            #Want to go from j2000t = j2000_current to j2000_start_planning - 1
            t = j2000_start_precalc
            actual_prop_status = self.prop_status
            while t<t_start_planning:
                ctrl_mode = self.goals.get_control_mode(t)
                # breakpoint()
                if ctrl_mode in PlannerModeList and self.current_trajectory.time_in_span(t):
                    #go to end of that trajectory and run whatever comes next.
                    x_prec = self.current_trajectory.last_state()
                    x_prec_prev = self.current_trajectory.penultimate_state()
                    tmax = self.current_trajectory.max_time()
                    if (tmax-t_start_planning)*cent2sec > -time_eps:
                        t = t_start_planning
                        x_prec = self.current_trajectory.state_nearest_to_time(t)
                        x_prec_prev = self.current_trajectory.state_nearest_to_time(t-self.time_step/cent2sec)
                    else:
                        t = tmax

                elif ctrl_mode in PlannerModeList and self.precalculation_ready and self.next_trajectory.time_in_span(t):
                    #go to end of that trajectory and run whatever comes next.
                    x_prec = self.next_trajectory.last_state()
                    x_prec_prev = self.next_trajectory.penultimate_state()
                    tmax = self.next_trajectory.max_time()
                    if (tmax-t_start_planning)*cent2sec > -time_eps:
                        t = t_start_planning
                        x_prec = self.next_trajectory.state_nearest_to_time(t)
                        x_prec_prev = self.next_trajectory.state_nearest_to_time(t-self.time_step/cent2sec)
                    else:
                        t = tmax
                else:

                    #If in plan and track but traj doesn't exist, do bdot
                    if ctrl_mode in PlannerModeList:
                        ctrl_mode = self.baseline_ctrl_mode
                    #Get B_ECI, r_ECI at t, t-1, and t+1
                    os = orbit_prec.get_os(t)
                    osp1 = orbit_prec.get_os(t+self.time_step/cent2sec)
                    # osm1 = orbit_prec.get_os(t-self.time_step/cent2sec)
                    #breakpoint()
                        #dB_body = dB_body#np.atleast_2d(np.array(np.matrix(((rot.T@Bt_ECI -rot_mat(x_prec_prev[-4:]).T@Bt_ECI)/self.sample_time)).A1)).reshape(3)
                    #Get control vector by simulating whatever mode we're in (not PLAN_AND_TRACK).
                    goal = self.goals.get_pointing_info(os,quatmode = self.quatvecmode)
                    goalp1 = self.goals.get_pointing_info(osp1,quatmode = self.quatvecmode)
                    ctrl_goal = ([],[],[],[],goal.eci_vec,goal.body_vec)
                    ctrl_goalp1 = ([],[],[],[],goalp1.eci_vec,goalp1.body_vec)

                    # prop_inds = np.where([j<t for j in prop_prec_t])[0]
                    # if len(prop_inds) == 0:
                    #     raise ValueError("There should always be a time in the list less than this...")
                    # prop_t = prop_prec_p[max(prop_inds)]
                    prec_law = [j for j in self.control_laws if j.modename==ctrl_mode][0]
                    u_prec = prec_law.find_actuation(self.estimator.use_state.val,self.orbit_estimator.current_state_estimate,self.next_os,self.current_goal,self.prev_goal,self.next_goal,np.nan,(ctrl_goal,ctrl_goalp1),True)
                    #Now, propagate attitude forward using rk4 + normalize quaternion
                    #Increment t
                    dt = min(abs(t_start_planning-t)*cent2sec,self.time_step)
                    x_prec_prev = x_prec
                    prop_j = prop_prec_p[np.where(np.array(prop_prec_t)<=t)[0][-1]]
                    use_prop_j = prop_j and self.planner_settings.plan_for_prop
                    self.prec_t += [t]
                    self.prec_X += [x_prec]
                    self.prec_U += [u_prec]
                    if use_prop_j:
                        self.virtual_sat.prop_dist_on()
                    else:
                        self.virtual_sat.prop_dist_off()
                    x_prec = self.virtual_sat.rk4(x_prec[0:self.virtual_sat.state_len], u_prec, gcd_dt_precalc, os,osp1)
                    #TODO: add disturbance estimation here?
                    t += dt/cent2sec

            if actual_prop_status:
                self.virtual_sat.prop_dist_on()
            else:
                self.virtual_sat.prop_dist_off()

        #Finally, calculate point_ECI, sat_vec and get matrix inputs ready for TP
        #Initialize matrix inputs for TP to all zeros
        # duration = math.ceil((j2000_end_planning-j2000_start_planning)/gcd_dt_altro)+1
        # tp_times = sorted(list(set([t_start_planning] + np.arange(t_start_planning,j2000_end_precalc,gcd_dt_tp).tolist() + [j2000_end_precalc])))

        #Get pointing goal vec, sat pointing vec, r_ECI, v_ECI, sun_ECI, B_ECI for t=j2000_start_planning...j2000_end_planning + 1
        # t_tp = t_start_planning
        #while j2000_delta(j2000_end_planning, vec_j2000) < 0.5:
        #[prop_prec_p[np.where(np.array(prop_prec_t)<j)] for j in tp_orbit.times if tp_orbit.time_in_span(prop_prec_t[j])]
        #TODO adjust x_prec for systems with RWs by using their estimated state.
        return x_prec

    def check_for_precalculate_and_do(self):
        """
        This function checks if we are ready to precalculate given the current time tuple.

        Parameters
        ------------
            current_time: tuple
                time in centuries since January 1, 2000 (in UTC time)

        Returns
        ---------
            precalculate: boolean
                True if we should start precalculating, False otherwise
        """
        # current_j2000 = self.current_j2000
        future_j2000 = self.current_j2000 + (self.planner_settings.precalculation_time)/cent2sec
        # ctrl_mode_now = self.goals.get_control_mode(current_j2000)
        ctrl_mode_future = self.goals.get_control_mode(future_j2000)

        # in_plan_and_track = (ctrl_mode_now == GovernorMode.PLAN_AND_TRACK_LQR or ctrl_mode_now == GovernorMode.PLAN_AND_TRACK_MPC) #TODO should this be if *any* time between then are planned?
        currently_planning = self.precalculation_ready

        if self.current_trajectory.is_empty():
            need_new_traj = (ctrl_mode_future in PlannerModeList)
            j2000_start_planning = self.goals.get_next_ctrlmode_change_time(self.current_j2000)
        else:
            farfuture_j2000 = (future_j2000+(self.planner_settings.traj_overlap-time_eps)/cent2sec)
            future_in_traj_overlap_region_from_traj_end = farfuture_j2000 >= self.current_trajectory.max_time()
            future_in_plannermode = ctrl_mode_future in PlannerModeList
            planning_continues_after_traj_end = self.goals.get_control_mode(farfuture_j2000) in PlannerModeList#no need to have overlap with ending of trajectory
            need_new_traj = future_in_traj_overlap_region_from_traj_end and future_in_plannermode and planning_continues_after_traj_end
            j2000_start_planning = self.current_trajectory.max_time()-self.planner_settings.traj_overlap/cent2sec

        need_to_precalculate = (not currently_planning) and need_new_traj
        # self.prev_future_j2000 =  future_j2000
        if not need_to_precalculate:
            return False,[]

        #Get start and end of planning
        """
        cntrl_switch, _, keys, _ = self.get_cntrl_switch(current_time)
        use_ind = np.where(cntrl_switch)[0][0]
        t_start_planning = self.control_change_times[keys[use_ind]]
        t_end_planning = self.control_change_times[keys[use_ind+1]]
        """
        j2000 = self.current_j2000
        j2000_end_planning_1 = j2000_start_planning + self.planner_settings.default_traj_length*1.0/cent2sec + self.planner_settings.traj_overlap*1.0/cent2sec
        j2000_end_planning_2 = self.goals.get_next_ctrlmode_change_time(j2000_start_planning)
        j2000_end_planning = min(j2000_end_planning_1,j2000_end_planning_2)
        #Call main precalculate fn to get tp inputs.
        x_prec = self.precalculate_for_trajectory_planner(j2000_start_planning, j2000_end_planning)
        #(bdot_on, plot_on) = self.tp_settings
        #TODO: Figure out if we need to do something with plan_for_prop here
        #Calculate N and t_final planning (different from t_end due to trajectory overlap)
        return need_to_precalculate,x_prec
        # return x_prec

    def switch_to_precalculated_trajectory(self, verbose=False, x_estimate=None):
        """
        This function switches an already-precalculated trajectory to be the "current" trajectory
        """
        if x_estimate is None:
            x_estimate = np.zeros(self.virtual_sat.state_len)
        # print(self.next_trajectory.gains)
        self.current_trajectory = self.next_trajectory.copy()
        self.next_trajectory = Trajectory()

        self.precalculation_ready = False
        ################################
        if verbose:
            print("\n\n********************\n********************\n"+
                            "estimated traj start: "+str(self.current_trajectory.first_state())+
                          "\n   actual traj start: "+str(x_estimate.T)+
                            "\n********************\n********************\n\n")

    def call_trajOpt_python_debug(self, planner, vecs_w_time, N, t_start, t_end, x0, bdotOn,axes,fig):
        # print('x0',x0[3:7].T)
        x0 = x0.astype(np.double)
        first_pass_no_dist = False
        # print('initial ECI vec',vecs_w_time[7][:,0].T,vecs_w_time[6][:,0].T,x0[3:7].T,np.arccos(np.dot(vecs_w_time[6][:,0],rot_mat(x0[3:7]).T@vecs_w_time[7][:,0]))*180.0/np.pi)
        # print( (vecs_w_time[0]-0.22)*cent2sec)
        (traj_init,vecs,costSettings) = planner.prepareForAlilqr(vecs_w_time, self.planner_settings.dt_tp, t_start, t_end, x0, bdotOn)
        # print((vecs[0]-0.22)*cent2sec)
        # print('initial ECI vec',vecs[7][:,0].T,vecs[6][:,0].T,traj_init[0][3:7,0].T,np.arccos(np.dot(vecs[6][:,0],rot_mat(traj_init[0][3:7,0]).T@vecs[7][:,0]))*180.0/np.pi)

        alilqrOut = self.alilqr_python_for_debug_plot(planner,traj_init,vecs,costSettings,self.planner_settings.mainAlilqrSettings(),axes,fig,not first_pass_no_dist)

        opt_tmp0 = alilqrOut[0];
        Xtmp0 = opt_tmp0[0];
        Utmp0 = opt_tmp0[1];
        anyDist = bool(self.planner_settings.plan_for_gg or self.planner_settings.plan_for_srp or self.planner_settings.plan_for_aero or self.planner_settings.plan_for_prop or self.planner_settings.plan_for_resdipole or self.planner_settings.plan_for_gendist);
        print("any dist?")
        if anyDist and first_pass_no_dist:
            print("doing dist")
            #if any disturbances, run the basic/larger-time-step alilqr again, with the disturbances on.
            traj_0 = (Xtmp0,Utmp0,opt_tmp0[5],opt_tmp0[2]);
            alilqrOut = self.alilqr_python_for_debug_plot(planner,traj_0, vecs, costSettings,self.planner_settings.mainAlilqrSettings(),axes,fig,True)
        # breakpoint()
        # alilqrOut = planner.alilqr(1*sec2cent, traj_init,  vecs, costSettings,  self.planner_settings.mainAlilqrSettings(), True)
        (_,vecs2,_) = planner.prepareForAlilqr(vecs_w_time, self.planner_settings.dt_tvlqr, t_start, t_end, x0, bdotOn)
        Rset_tvlqr = vecs2[1]
        if(self.planner_settings.dt_tp/self.planner_settings.dt_tvlqr > 1):
            opt_tmp1 = alilqrOut[0];
            Xtmp1 = opt_tmp1[0];
            Utmp1 = opt_tmp1[1];
            colMissing = max(0,Rset_tvlqr.shape[1] - 1 - (Utmp1.shape[1]-2)*int(self.planner_settings.dt_tp/self.planner_settings.dt_tvlqr))
            # breakpoint()
            UsetLong = np.concatenate([np.repeat(Utmp1[:,:Utmp1.shape[1]-2],int(self.planner_settings.dt_tp/self.planner_settings.dt_tvlqr),axis = 1)]+[np.repeat(Utmp1[:,Utmp1.shape[1]-2:Utmp1.shape[1]-1],colMissing,axis = 1)]+[Utmp1[:,Utmp1.shape[1]-1:Utmp1.shape[1]]],axis = 1)
            # breakpoint()
            trajLong = planner.generateInitialTrajectory(self.planner_settings.dt_tvlqr,np.copy(Xtmp1[:,0]), UsetLong, vecs2); #really only done to get the time vec correct ugh
            alilqrOut = self.alilqr_python_for_debug_plot(planner,trajLong, vecs2, self.planner_settings.optSecondCostSettings(),self.planner_settings.secondAlilqrSettings(),axes,fig,True)
        #(optOut,mu,lastGrad) = alilqrOut
        (_, _,_,lqr_opt,_) = planner.cleanUpAfterAlilqr(vecs_w_time, self.planner_settings.dt_tvlqr, t_start, t_end, alilqrOut)
        #(Xset, Uset, Kset, lamset) = main_opt
        (Xset_lqr, Uset_lqr,Tset_lqr, Kset_lqr, Sset_lqr,lqr_times) = lqr_opt
        return lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,Tset_lqr


    def ilqr_step_plot(self,planner,dt0,traj,vecs,auglag_vals,regs,costSettings,regSettings,lineSearchSettings, breakSettings, useDist,fig,axes):
        N = traj[0].shape[1]
        (BPresults,regs) = planner.backwardPass(dt0,traj,vecs,auglag_vals,regs,costSettings,regSettings,useDist);


        plt.figure(fig.number)
        axes[2].clear()
        axes[2].cla()
        axes[2].plot(BPresults[1].T)
        axes[2].set_ylabel('dset')
        (traj,newLA,regs) = planner.forwardPass(dt0,traj,vecs,auglag_vals,BPresults,regs,costSettings,regSettings,lineSearchSettings,useDist);
        # newLAnc = planner.cost2Func(traj, vecs, auglag_vals, costSettings,False);
        grad = sum( np.max(np.abs(BPresults[1][:,0:N-2])/(np.abs(traj[1][:,0:N-2] )+1),0))/(N-1);
        (clist,cmaxtmp) = planner.maxViol(traj,vecs,auglag_vals);
        return (newLA,cmaxtmp,clist,grad,regs,traj);



    def alilqr_python_for_debug_plot(self,planner,traj,vecs,costSettings_tmp,alilqrSettings_tmp,axes,fig,has_dist = False):

        isFirstSearch = True

        self.planner_settings.fig_exists = False
        lineSearchSettings_tmp = alilqrSettings_tmp[0]
        auglagSettings_tmp = alilqrSettings_tmp[1]
        breakSettings_tmp = alilqrSettings_tmp[2]
        regSettings_tmp = alilqrSettings_tmp[3]
        lagMultInit_tmp = auglagSettings_tmp[0]
        penInit_tmp = auglagSettings_tmp[2]
        penScale_tmp = auglagSettings_tmp[4]
        maxCost_tmp = breakSettings_tmp[8]
        maxOuterIter_tmp = breakSettings_tmp[0]
        maxIlqrIter_tmp = breakSettings_tmp[1]
        zCountLim_tmp = breakSettings_tmp[6]
        cmax_tmp = breakSettings_tmp[7]
        regInit_tmp = regSettings_tmp[0]


        Xset = traj[0];
        # print(Xset[3:7,0].T)
        Uset = traj[1];
        TQset = traj[3];
        dt_timevec = traj[2]
        dt0 = (dt_timevec[1]-dt_timevec[0])*cent2sec
        # print('initial ECI vec at alilqr beginning',vecs[7][:,0].T,vecs[6][:,0].T,Xset[3:7,0].T,np.arccos(np.dot(vecs[6][:,0],rot_mat(Xset[3:7,0]).T@vecs[7][:,0]))*180.0/np.pi)


        traj = planner.generateInitialTrajectory(dt0,np.copy( Xset[:,0]), Uset,vecs)


        Xset = traj[0];
        Uset = traj[1];
        TQset = traj[3];
        dt_timevec = traj[2]
        dt0 = (dt_timevec[1]-dt_timevec[0])*cent2sec


        N = Xset.shape[1]
        constraint_N = int(self.planner_settings.wmax>0) + int(self.planner_settings.sun_limit_angle>0) + 2*self.virtual_sat.control_len + 3*self.virtual_sat.number_RW
        #initialize grad, iter, lambdaSet, mu, LA0
        grad = 1.0/num_eps
        iter = 0
        lambdaSet = np.ones((constraint_N,N))*lagMultInit_tmp;
        muSet = np.ones((constraint_N,N))*penInit_tmp/penScale_tmp;
        mu = penInit_tmp/penScale_tmp;
        auglag_vals = (lambdaSet,mu,muSet)
        viol = planner.maxViol(traj,vecs,auglag_vals);
        clist = viol[0];
        auglag_vals = planner.incrementAugLag(auglag_vals,clist,auglagSettings_tmp);
        auglag_vals_clean = (0*lambdaSet,0,0*muSet);
        LA0 = planner.cost2Func(traj, vecs, auglag_vals, costSettings_tmp);
        LA = LA0;
        LAnc = planner.cost2Func(traj, vecs, auglag_vals_clean, costSettings_tmp);


        cmaxtmp = 0.0;
        dlaZcount = 0;
        dLA = 0.0;
        newLA = LA;
        regs = (copy.deepcopy(regSettings_tmp[0]),0);
        prevU = Uset*0;

        # stepsSinceRand = -1;
        #initialize Kset, Pset
        # BACKWARD_PASS_RESULTS_FORM BPresults;
        # tuple<double,double,mat,double,REG_PAIR,TRAJECTORY_FORM> ilqrRes;
        #Outer loop


        if not self.planner_settings.fig_exists:
            ffig = plt.figure(1)
            ffig.clf()
            aax0 = ffig.add_subplot(812)
            aax0.cla()
            aax1 = ffig.add_subplot(813)
            aax1.cla()
            aax2 = ffig.add_subplot(814)
            aax2.cla()
            aax3 = ffig.add_subplot(815)
            aax3.cla()
            aax4 = ffig.add_subplot(816)
            aax4.cla()
            aax5 = ffig.add_subplot(817)
            aax5.cla()
            aax6 = ffig.add_subplot(818)
            aax6.cla()


            tvec = (traj[2]-0.22)*cent2sec
            aax0.clear()
            aax0.cla()
            aax0.set_title("starter!"+"has dist? "+str(has_dist))
            aax0.plot(tvec,Xset[0:3,:].T)
            aax0.set_ylabel('AV')
            aax1.clear()
            aax1.cla()
            aax1.plot(tvec,Uset.T)
            aax1.set_ylabel('Ctrl')
            aax2.clear()
            aax3.cla()
            aax2.plot(tvec,Xset[3:7,:].T)
            aax2.set_ylabel('Quat')
            pt_err = np.arccos(np.array([np.dot(vecs[6][:,j],rot_mat(Xset[3:7,j]).T@vecs[7][:,j]) for j in range(Xset.shape[1])]))*180.0/np.pi

            aax3.clear()
            aax3.cla()
            aax3.plot(tvec,pt_err)
            aax3.set_ylabel('ctrlang')

            aax4.clear()
            aax4.cla()
            aax4.plot(tvec,vecs[7][:,:Xset.shape[1]].T)
            aax4.set_ylabel('ECI Vec')

            eci_vec_body = np.stack([rot_mat(Xset[3:7,j]).T@vecs[7][:,j] for j in range(Xset.shape[1])])
            aax5.clear()
            aax5.cla()
            aax5.plot(tvec,eci_vec_body)
            aax5.set_ylabel('ECI Vec in Body')


            aax6.clear()
            aax6.cla()
            aax6.plot(tvec,TQset.T)
            aax6.set_ylabel('torq')
            aax6.set_xlabel('time')

            # aax-1].clear()
            # aax-1].cla()
            # aax-1].plot(tvec,pt_err)
            # aax-1].set_ylabel('ctrlang')
            # aax-1].set_xlabel('time')
            plt.draw()
            plt.pause(0.05)
            self.planner_settings.fig_exists = True
        print('begin outer')
        for j in range(maxOuterIter_tmp):
            # print("outer iter",j)
            #reset cmaxtmp, dlaZcount
            cmaxtmp = 0.0;
            dlaZcount = 0;
            clist = clist*0;
            dLA = 0.0;

            # stepsSinceRand = -1;
            #ILQR
            #set rho and drho to regInit
            regs = (copy.deepcopy(regSettings_tmp[0]),0);
            #Find initial cost and init dLA
            LA = planner.cost2Func(traj,vecs, auglag_vals, costSettings_tmp);
            #inner loop
            for ii in range(maxIlqrIter_tmp):
                #update iter
                # print("ii: ",ii)
                iter += 1

                # rnOut = planner.addRandNoise(dt0, traj,  dlaZcount,  stepsSinceRand, breakSettings_tmp, regSettings_tmp, costSettings_tmp, auglag_vals, vecs);
                # traj = rnOut[0]
                # stepsSinceRand = rnOut[1]
                ilqrRes = planner.ilqrStep(dt0,traj,vecs,auglag_vals,regs,costSettings_tmp,regSettings_tmp,lineSearchSettings_tmp,breakSettings_tmp,has_dist);
                # ilqrRes = self.ilqr_step_plot(planner,dt0,traj,vecs,auglag_vals,regs,costSettings_tmp,regSettings_tmp,lineSearchSettings_tmp,breakSettings_tmp,has_dist,fig,axes);

                (newLA,cmaxtmp,clist,grad,regs,traj) = ilqrRes
                dLA = np.abs(newLA-LA);
                dlaZcount += 1
                # if stepsSinceRand != 0:
                if dLA != 0:
                    dlaZcount = 0;
                # print("DLA INFO",dLA,dlaZcount)
                #update LA, Xset, Uset
                # stepsSinceRand += (stepsSinceRand>=0)
                LA = newLA;
                LAnc = planner.cost2Func(traj,vecs, auglag_vals_clean, costSettings_tmp);
                current_traj = traj;
                current_Xset = traj[0];
                current_Uset = traj[1];
                current_TQset = traj[3];
                # print(np.max(np.abs(current_TQset)))
                current_ilqr_iter = ii;
                current_outer_iter = j;
                # print(current_Xset[3:7,0].T)
                # print(ii,j," cmaxtmp,LA,LA clean ",cmaxtmp,LA,LAnc)
                #Check if we need to break out of the loop
                # print((traj[2]-0.22)*cent2sec)
                tvec = (traj[2]-0.22)*cent2sec
                plt.figure(fig.number)
                axes[0].clear()
                axes[0].cla()
                axes[0].set_title("outer: "+str(j)+" inner: "+str(ii)+"has dist? "+str(int(has_dist))+"reg: "+str(regs[0])+"LA: "+str(newLA))
                axes[0].plot(tvec,current_Xset[0:3,:].T)
                axes[0].set_ylabel('AV')
                axes[1].clear()
                axes[1].cla()
                axes[1].plot(tvec,current_Uset.T)
                axes[1].set_ylabel('Ctrl')
                axes[2].clear()
                axes[2].cla()
                axes[2].plot(tvec,vecs[7].T)
                axes[2].set_ylabel('ECIvec')
                pt_err = np.arccos(np.array([np.dot(vecs[6][:,j],rot_mat(current_Xset[3:7,j]).T@vecs[7][:,j]) for j in range(current_Xset.shape[1])]))*180.0/np.pi

                axes[3].clear()
                axes[3].cla()
                axes[3].plot(tvec,pt_err)
                axes[3].set_ylabel('ctrlang')

                axes[4].clear()
                axes[4].cla()
                axes[4].plot(tvec,current_TQset.T)
                axes[4].set_ylabel('dist torq')

                eci_vec_body = np.stack([rot_mat(current_Xset[3:7,j]).T@vecs[7][:,j] for j in range(current_Xset.shape[1])])
                axes[5].clear()
                axes[5].cla()
                axes[5].plot(tvec,eci_vec_body)
                axes[5].set_ylabel('ECI Vec in Body')


                axes[6].clear()
                axes[6].cla()
                axes[6].plot(tvec,(current_Uset-prevU).T)
                axes[6].set_ylabel('delU')
                axes[6].set_xlabel('time')

                prevU = np.copy(current_Uset)


                # aax-1].clear()
                # aax-1].cla()
                # aax-1].plot(tvec,pt_err)
                # aax-1].set_ylabel('ctrlang')
                # aax-1].set_xlabel('time')
                plt.draw()
                plt.pause(0.05)
                if planner.ilqrBreak(grad,LA,dLA,dlaZcount,cmaxtmp,iter,breakSettings_tmp,j,ii,False):
                    break;
            if planner.outerBreak(auglag_vals,cmaxtmp,breakSettings_tmp,auglagSettings_tmp,j) and j>2 and planner.ilqrBreak(grad,LA,dLA,dlaZcount,cmaxtmp,iter,breakSettings_tmp,j,ii,True) :
              # print("outerbreak")
              break;

            auglag_vals = planner.incrementAugLag(auglag_vals,clist,auglagSettings_tmp);
        # mat Kmat = packageK(get<0>(BPresults));
        opt = (traj[0],traj[1],traj[3],[],lambdaSet,dt_timevec);
        return (opt, mu, grad)

    # @staticmethod


    def collocation_constraints(self,state_vars, control_vars, h, num_samples,vecs):
        constraints = []

        for i in range(num_samples - 1):
            # Get state and control at two consecutive collocation points
            x0 = state_vars[i]
            x1 = state_vars[i + 1]
            u0 = control_vars[i]
            u1 = control_vars[i + 1]

            os0 = Orbital_State(t=vecs[0][i],R=vecs[1][:,i],V=vecs[2][:,i],B=vecs[3][:,i],S=vecs[4][:,i],rho = vecs[5][i])
            os1 = Orbital_State(t=vecs[0][i+1],R=vecs[1][:,i+1],V=vecs[2][:,i+1],B=vecs[3][:,i+1],S=vecs[4][:,i+1],rho = vecs[5][i])

            # Dynamics at the collocation points (using trapezoidal rule)
            f0 = self.virtual_sat.noiseless_dynamics(x0,u0,os0)
            f1 = self.virtual_sat.noiseless_dynamics(x1,u1,os1)
            xmid = 0.5*(x0+x1) + (h/8)*(f1-f0)
            xmid[3:7] = normalize(xmid[3:7])
            fmid = self.virtual_sat.noiseless_dynamics(xmid,0.5*(u0+u1),os0.average(os1))

            constraints.append(fmid + (1.5/h)*(x0-x1) + 0.25*(f0+f1))

        return np.concatenate(constraints)

    def extra_constraints(self,state_vars, control_vars, h, num_samples,vecs):
        constraints = []

        for i in range(num_samples - 1):
            # Get state and control at two consecutive collocation points
            x0 = state_vars[i]
            x1 = state_vars[i + 1]
            u0 = control_vars[i]
            u1 = control_vars[i + 1]

            constraints.append(norm(x1[3:7])-1)

        return np.array(constraints)


    def objective_function(self,controls,state,vecs, num_samples):
        control_cost = 0
        pt_cost = 0
        for u in controls:
            control_cost += np.dot(u, u)*self.planner_settings.u_weight_mult*self.planner_settings.mtq_control_weight  # Sum of squares of torques

        for t in range(len(vecs[0])):
            pt_cost += (1-np.dot(vecs[6][:,t],rot_mat(state[t,3:7]).T@vecs[7][:,t]))*self.planner_settings.angle_weight

        return control_cost+pt_cost

    def call_outside_trajOpt(self,planner, vecs_w_time, N, t_start, t_end, x0, bdotOn):

        if self.outside_solver == OUTSIDE_SOLVERS.SciPy:

            (traj_init,vecs,costSettings) = planner.prepareForAlilqr(vecs_w_time, self.planner_settings.dt_tp, t_start, t_end, x0, bdotOn)
            # Create the optimization problem
            # prog = MathematicalProgram()

            # Define time span and number of collocation points
            num_samples = vecs[0].shape[0] # Number of collocation points
            duration = (t_end-t_start)*cent2sec  # Duration of the trajectory

            # Create decision variables for time, state, and input
            h = self.planner_settings.dt_tp  # Time step (you can make it variable too)
            # Initialize decision variables for the state (ω, q) and control (τ)
            state_dim = 7  # 3 for angular velocity + 4 for quaternion
            control_dim = 3  # 3 torques

            h = duration / (num_samples - 1)  # Time step

            # Initialize decision variables (flattened form of states and controls)
            state_guess = np.zeros((num_samples, 7))  # Initial guess for state trajectory
            control_guess = np.zeros((num_samples, 3))  # Initial guess for control trajectory

            # Flatten the initial guesses for the optimizer
            initial_guess = np.concatenate([state_guess.flatten(), control_guess.flatten()])

            # Constraint function for the optimizer
            def constraints_flat(vars_flat):
                # Extract state and control variables from the flattened array
                state_vars = vars_flat[:num_samples * 7].reshape((num_samples, 7))
                control_vars = vars_flat[num_samples * 7:].reshape((num_samples, 3))

                # Boundary conditions (initial and final states)
                state_constraints = []
                state_constraints.append(state_vars[0] - x0[:7])  # Initial state constraint

                # Add collocation constraints
                collocation_constr = self.collocation_constraints(state_vars, control_vars, h, num_samples,vecs)
                extra_constr = self.extra_constraints(state_vars, control_vars, h, num_samples,vecs)

                return np.concatenate(state_constraints + [collocation_constr]+[extra_constr])

            # Objective function for the optimizer (minimize control effort)
            def objective_flat(vars_flat):
                control_vars = vars_flat[num_samples * 7:].reshape((num_samples, 3))
                state_vars = vars_flat[:num_samples * 7].reshape((num_samples, 7))
                return self.objective_function(control_vars,state_vars,vecs, num_samples)


            control_bounds = [(-self.planner_settings.umax[i], self.planner_settings.umax[i]) for _ in range(num_samples) for i in range(3)]

            # Define bounds for all decision variables (states have no bounds, only controls)
            bounds = [(None, None)] * (num_samples * 7) + control_bounds


            # Run optimization
            result = scipy.optimize.minimize(objective_flat, initial_guess, constraints={'type': 'eq', 'fun': constraints_flat},bounds=bounds)

            if result.success:
                # Extract optimal solution
                optimal_vars = result.x
                optimal_state = optimal_vars[:num_samples * 7].reshape((num_samples, 7))
                optimal_control = optimal_vars[num_samples * 7:].reshape((num_samples, 3))
                opt = (optimal_state,optimal_control,[],[],[],vecs[0])
                return (opt, [], [])
            else:
                raise ValueError("Optimization failed!")


        else:
            raise ValueError("wrong solver")


    # @staticmethod
    def call_trajOpt_cpp(self,planner, vecs_w_time, N, t_start, t_end, x0, bdotOn):

        #Not getting success, lastGrad, main_opt
        print(x0)
        # print(x0.dtype)
        # print(t_start,t_end)
        # x0 = np.array([j*1.0 for j in list(x0)])
        x0 = x0.astype(np.double)
        # print(x0.dtype)

        (_, _,_,lqr_opt, _) = planner.trajOpt(vecs_w_time, N, t_start, t_end, x0, bdotOn)
        (Xset_lqr, Uset_lqr,Tset_lqr, Kset_lqr, Sset_lqr,lqr_times) = lqr_opt

        print(Xset_lqr[:,0])
        print(x0)
        return lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,Tset_lqr

    def live_debug_plot(self, fig, axes, planner_args):
        (planner, vecsPy, testN,j1, j2, x0, bdotOn) = planner_args
        lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,Tset_lqr = self.call_trajOpt_python_debug(planner, vecsPy, testN, j1, j2, x0, bdotOn,axes,fig)
        return lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,Tset_lqr

    def start_trajectory_planning(self, x0, debug_plot_on=None):

        bdotOn = self.planner_settings.bdot_on
        if debug_plot_on is None:
            debug_plot_on=self.planner_settings.debug_plot_on
        planner = self.planner
        print("%%%%%%%%%%%%%%%%",(np.array(self.tp_orbit.times).min()-0.22)*cent2sec)
        vecsPy = tuple([np.copy(np.array(self.tp_orbit.times), order='C')]+[np.copy(np.squeeze(np.array(k)).T, order='C') for k in self.tp_orbit.get_vecs()] + [np.copy(np.squeeze(self.tp_sat_vec).T, order='C')]+[np.copy(np.squeeze(self.tp_point_goal).T, order='C')]+[np.copy(np.array(self.tp_prop), order='C')])
        N = len(self.tp_orbit.times)
        t_start_planning = self.tp_orbit.min_time()
        t_end_planning = self.tp_orbit.max_time()
        if debug_plot_on:
            #This will pause sim
            planner_args = (self.planner, vecsPy, N, t_start_planning, t_end_planning, x0, bdotOn)
            fig = plt.figure(2)
            fig.clf()
            plt.clf()
            ax = fig.add_subplot(812)
            ax.cla()
            ax2 = fig.add_subplot(813)
            ax2.cla()
            ax3 = fig.add_subplot(814)
            ax3.cla()
            ax4 = fig.add_subplot(815)
            ax4.cla()
            ax5 = fig.add_subplot(816)
            ax5.cla()
            ax6 = fig.add_subplot(817)
            ax6.cla()
            ax7 = fig.add_subplot(818)
            ax7.cla()
            axes = (ax,ax2,ax3,ax4,ax5,ax6,ax7)
            lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,Tset_lqr = self.live_debug_plot(fig, axes, planner_args)
            return lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,Tset_lqr
        else:
            # lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,Tset_lqr = self.call_outside_trajOpt(planner, vecsPy, N, t_start_planning, t_end_planning, x0, bdotOn)
            lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,Tset_lqr = self.call_trajOpt_cpp(planner, vecsPy, N, t_start_planning, t_end_planning, x0, bdotOn)

            return lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,Tset_lqr
