import numpy as np
import scipy
import scipy.linalg
import random
import math
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
import pytest
from sat_ADCS_satellite import *
import warnings
from .attitude_estimator import *

class PerfectEstimator(Estimator):
    def __init__(self, j2000,sat,estimate,cov_estimate,integration_cov,sample_time = 1,use_cross_term = False,quat_as_vec = True,adj_kap_by_L = False,sunsensors_during_eclipse = False,verbose = False):
        super().__init__(j2000,sat,estimate,cov_estimate,integration_cov,sample_time,use_cross_term,quat_as_vec,sunsensors_during_eclipse = sunsensors_during_eclipse,verbose = verbose)

    def update(self,control_vec,sensors_in,os,which_sensors = None,truth = None):
        if truth is None:
            error("can't be None")
        if self.has_sun and os.in_eclipse():
            self.byebye_sun()
        if not self.has_sun and not os.in_eclipse():
            self.hello_sun()
        if self.prev_os.R.all() == 0:
            self.prev_os = os
        #     self.prev_os_vecs = os_local_vecs(os,self.use_state.val[3:7])
        # else:
        #     self.prev_os_vecs = self.os_vecs.copy()
        if which_sensors is None:
            which_sensors = [True for j in self.sat.attitude_sensors]
        if not self.has_sun and not self.sunsensors_during_eclipse:
            which_sensors = [which_sensors[j] and not isinstance(self.sat.attitude_sensors[j],SunSensor) for j in range(len(which_sensors))]
            which_sensors = [which_sensors[j] and not isinstance(self.sat.attitude_sensors[j],SunSensorPair) for j in range(len(which_sensors))]
        else: # has sun, ignore readings from sun sensors that are too little--could be albedo, noise, etc.
            which_sensors = [which_sensors[j] and not (isinstance(self.sat.attitude_sensors[j],SunSensor) and ((sensors_in[j] - self.sat.attitude_sensors[j].bias) < self.sat.attitude_sensors[j].std))  for j in range(len(which_sensors))]
        # # print(which_sensors)


        # scalar_update = False
        truth_cov = np.eye(self.sat.state_len-1+self.quat_as_vec)*1e-20
        #remove values not in use
        pv = np.concatenate([truth,self.use_state.val[self.sat.state_len:]])
        pc = scipy.linalg.block_diag(truth_cov, self.use_state.cov[self.sat.state_len-1:,self.sat.state_len-1:])
        #TODO: add a predictive step so dynamics are captured and this thing will actually pick up disturbances and actuator biases--was attempted but ran into issues that I think are numerical. Perhaps two corrective steps--one with truth for dynamics, then one with sensors for sensor biases?



        predicted_state = self.predictive_step(os,control_vec)
        # #remove values not in use
        pv = predicted_state.val
        pc = predicted_state.cov
        # breakpoint()


        vecs = os_local_vecs(os,pv[3:7])
        for j in range(len(self.sat.sensors)): #ignore sun sensors that should be in shadow
            if isinstance(self.sat.sensors[j],SunSensor):
                which_sensors[j] &= not (self.sat.sensors[j].clean_reading(pv,vecs)<1e-10)
        print(which_sensors)

        state_mod = np.eye(self.sat.state_len)
        if not self.quat_as_vec:
            state_mod = block_diag(np.eye(3),vec3_to_quat_deriv(quat_to_vec3(pv[3:7],self.vec_mode),self.vec_mode),np.eye(self.sat.state_len-7))

        # if scalar_update:
        #     state2,cov2 = self.scalar_update(pc,pv,os)
        # else:
            #sensor jacobian
        state_jac,bias_jac = self.sat.sensor_state_jacobian(pv,vecs,which = which_sensors,keep_unused_biases=True)
        lenA = np.size(state_jac,1)
        Hk = np.vstack([state_mod@state_jac,np.zeros((self.sat.act_bias_len,lenA)),bias_jac,np.zeros((self.sat.dist_param_len-(len(self.use)-sum(self.use)),lenA))])
        sens_cov = self.sat.sensor_cov(which_sensors,keep_unused_biases=False)
        lilhk = self.sat.sensor_values(pv,vecs,which = which_sensors)
        zk = np.concatenate([sensors_in[0:len(which_sensors)][which_sensors],sensors_in[len(which_sensors):]])

        if not self.quat_as_vec:
            # breakpoint()
            # lilhk = np.concatenate([lilhk,predicted_state[:self.sat.state_len]])
            lilhk = np.concatenate([lilhk,pv[:3],np.zeros(3),pv[7:self.sat.state_len]])
            # zk = np.concatenate([zk,truth])
            zk = np.concatenate([zk,truth[:3],quat_to_vec3(quat_mult(quat_inv(pv[3:7]),truth[3:7]),self.vec_mode),truth[7:self.sat.state_len]])
            # state_mod2 = block_diag(np.eye(3),vec3_to_quat_deriv(quat_to_vec3(truth[3:7],self.vec_mode),self.vec_mode),np.eye(self.sat.state_len-7))
            # state_mod2 = block_diag(np.eye(3),vec3_to_quat_deriv(quat_to_vec3(quat_mult(quat_inv(truth[3:7]),pv[3:7]),self.vec_mode),self.vec_mode),np.eye(self.sat.state_len-7))

            # addmat = np.block([[state_mod@np.eye(self.sat.state_len)@state_mod2.T],[np.zeros((len(pv)-self.sat.state_len,self.sat.state_len-1+self.quat_as_vec))]])
            addmat = np.block([[np.eye(self.sat.state_len-1+self.quat_as_vec)],[np.zeros((len(pv)-self.sat.state_len,self.sat.state_len-1+self.quat_as_vec))]])

            Hk = np.block([[Hk,addmat]])
            sens_cov = scipy.linalg.block_diag(sens_cov,truth_cov)
        else:
            raise ValueError("the quat as vec/not thing needs to figured out at least for this estimator (perfect estimator)")


        try:

            Kk = scipy.linalg.solve((sens_cov+Hk.T@pc@Hk),(pc@Hk).T,assume_a='pos')
        except:
            # breakpoint()
            raise np.linalg.LinAlgError('Matrix is singular. (probably)')

        dstate = (zk-lilhk)@Kk
        cov2 = pc@(np.eye(len(pv) - 1 + self.quat_as_vec)-Hk@Kk)
        cov2 = 0.5*(cov2 + cov2.T)



        if self.quat_as_vec:
            state2 = pv + dstate
            state20 = np.copy(state2)
            state2[3:7] = normalize(state2[3:7])
            norm_jac = state_norm_jac(state20)
            cov2 = norm_jac.T@cov2@norm_jac
        else:
            state2 = pv.copy()
            state2[0:3] += dstate[0:3]
            state2[7:] += dstate[6:]
            state2[3:7] = quat_mult(state2[3:7],vec3_to_quat(dstate[3:6],self.vec_mode))

        out = estimated_nparray(state2,cov2)



        self.prev_os = os

        oc = out.cov
        # breakpoint()
        if not self.use_cross_term:
            # p = self.sat.state_len+self.sat.att_sens_bias_len+self.sat.act_bias_len - 1 + self.quat_as_vec
            # oc[0:p,p:] = 0
            # oc[p:,0:p] = 0
            ab0 = self.sat.state_len - 1 + self.quat_as_vec
            ab1 = self.sat.state_len - 1 + self.quat_as_vec + self.sat.act_bias_len
            sb0,sb1 = ab0 + self.sat.att_sens_bias_len,ab1 + self.sat.att_sens_bias_len
            d0 = sb1
            # oc[sb0:sb1,sb0:sb1] += np.diagflat(np.sum(oc[ab0:ab1,sb0:sb1],axis = 0))
            # oc[sb0:sb1,sb0:sb1] += np.diagflat(np.sum(oc[d0:,sb0:sb1],axis = 0))
            # oc[ab0:ab1,ab0:ab1] += np.diagflat(np.sum(oc[ab0:ab1,sb0:sb1],axis = 1))
            # oc[d0:,d0:] += np.diagflat(np.sum(oc[d0:,sb0:sb1],axis = 1))
            # oc[d0:,d0:] += np.diagflat(np.sum(oc[d0:,ab0:ab1],axis = 1))
            # oc[ab0:ab1,ab0:ab1] += np.diagflat(np.sum(oc[d0:,ab0:ab1],axis = 0))
            oc[ab0:ab1,sb0:sb1] = 0
            oc[sb0:sb1,ab0:ab1] = 0
            oc[ab0:ab1,d0:] = 0
            oc[d0:,ab0:ab1] = 0
            oc[sb0:sb1,d0:] = 0
            oc[d0:,sb0:sb1] = 0
        # breakpoint()
        self.full_state.set_indices(self.use,out.val,oc,square_mat_sections(self.full_state.int_cov,self.cov_use()),[3]*(not self.quat_as_vec))
        self.use_state = self.full_state.pull_indices(self.use,[3]*(not self.quat_as_vec))
        self.sat.match_estimate(self.full_state,self.update_period)
        # self.os_vecs = os_local_vecs(os,self.full_state.val[3:7])

        return self.full_state.val[0:self.sat.state_len],extra

        # if self.has_sun and os.in_eclipse():
        #     self.byebye_sun()
        # if not self.has_sun and not os.in_eclipse():
        #     self.hello_sun()
        # if self.prev_os.R.all() == 0:
        #     self.prev_os = os
        #     self.prev_os_vecs = os_local_vecs(os,self.full_state.val[3:7])
        # else:
        #     self.prev_os_vecs = self.os_vecs.copy()
        # if which_sensors is None:
        #     which_sensors = [True for j in self.sat.attitude_sensors]
        # if not self.has_sun and not self.sunsensors_during_eclipse:
        #     which_sensors = [which_sensors[j] and not isinstance(self.sat.attitude_sensors[j],SunSensor) for j in range(len(which_sensors))]
        #     which_sensors = [which_sensors[j] and not isinstance(self.sat.attitude_sensors[j],SunSensorPair) for j in range(len(which_sensors))]
        # else: # has sun, ignore readings from sun sensors that are too little--could be albedo, noise, etc.
        #     which_sensors = [which_sensors[j] and not (isinstance(self.sat.attitude_sensors[j],SunSensor) and ((sensors_in[j] - self.sat.attitude_sensors[j].bias) < self.sat.attitude_sensors[j].std))  for j in range(len(which_sensors))]
        # # # print(which_sensors)
        #
        #
        #
        # if self.prev_os.R.all() == 0:
        #     self.prev_os = os
        # truth_cov = np.eye(self.sat.state_len)*1e-25
        # # breakpoint()
        #
        # x1 = np.concatenate([truth,self.full_state.val.copy()[self.sat.state_len:]])
        # cov1 = self.full_state.cov.copy()+self.full_state.int_cov.copy()
        #
        # state_mod = np.eye(7)
        # # if not self.quat_as_vec:
        # #     state_mod = block_diag(np.eye(3),vec3_to_quat_deriv(quat_to_vec3(x1[3:7],self.vec_mode),self.vec_mode))
        # # print(len(self.sat.sensors),len(which_sensors))
        # numGPS = sum([isinstance(j,GPS) for j in self.sat.sensors])
        # # tmp = [self.sat.sensors[j].no_noise_reading(x1[0:self.sat.state_len],os_local_vecs(os,truth[3:7])) for j in range(len(self.sat.sensors)-numGPS) if which_sensors[j] and not isinstance(self.sat.sensors[j],GPS)]
        # # print(sensors_in.shape,sensors_in)
        # # print(tmp)
        # # lilhk = np.concatenate([np.array(sensors_in)]+[self.sat.sensors[j].no_noise_reading(x1[0:self.sat.state_len],os_local_vecs(os,truth[3:7])) for j in range(len(self.sat.sensors)-numGPS) if which_sensors[j] and not isinstance(self.sat.sensors[j],GPS)])
        # vecs = os_local_vecs(os,x1[3:7])
        #
        # lilhk = self.sat.sensor_values(x1,vecs,which = which_sensors)
        # sens_cov = self.sat.sensor_cov(which_sensors)
        #
        # state_jac,bias_jac = self.sat.sensor_state_jacobian(x1,vecs,which = which_sensors)
        # lenA = np.size(state_jac,1)
        # Hk = np.vstack([state_mod@state_jac,np.zeros((self.sat.act_bias_len,lenA)),bias_jac,np.zeros((len(self.use)-sum(self.use),lenA))])
        # zk = sensors_in
        #
        # if truth is not None and truth_cov is not None:
        #     lilhk = np.concatenate([lilhk,x1[0:self.sat.state_len]])
        #     breakpoint()
        #     Hk = np.block([[Hk,np.block([[np.eye(self.sat.state_len )],[np.zeros((len(x1)-self.sat.state_len ,self.sat.state_len ))]])]])
        #     sens_cov = scipy.linalg.block_diag(sens_cov,truth_cov)
        #     zk = np.concatenate([zk,truth])
        #
        # try:
        #     # print(np.linalg.cond(sens_cov_cut),np.linalg.cond(cov1_cut),np.linalg.cond(Hk_cut@cov1_cut@Hk_cut.T),np.linalg.cond((sens_cov_cut+Hk_cut@cov1_cut@Hk_cut.T)))
        #     Kk = scipy.linalg.solve((sens_cov+Hk.T@cov1@Hk),cov1@Hk,assume_a='pos')
        #     #Kk = (cov1@Hk1.T)@np.linalg.inv(sens_cov1+Hk1@cov1@Hk1.T)
        # except:
        #     raise np.linalg.LinAlgError('Matrix is singular. (probably)')
        #
        #
        # dstate = (sensors_in-lilhk)@Kk
        # cov2 = cov1@(np.eye(len(pv) - 1 + self.quat_as_vec)-Hk@Kk)
        # cov2 = 0.5*(cov2 + cov2.T)
        #
        # if self.quat_as_vec:
        #     state2 = x1 + dstate
        #     state20 = np.copy(state2)
        #     state2[3:7] = normalize(state2[3:7])
        #     norm_jac = state_norm_jac(state20)
        #     cov2 = norm_jac.T@cov2@norm_jac
        # else:
        #     state2 = x1.copy()
        #     state2[0:3] += dstate[0:3]
        #     state2[7:] += dstate[6:]
        #     state2[3:7] = quat_mult(state2[3:7],vec3_to_quat(dstate[3:6],self.vec_mode))
        #
        # out = estimated_nparray(state2,cov2)
        # self.prev_os = os
        #
        # oc = out.cov
        # if not self.use_cross_term:
        #     # p = self.sat.state_len+self.sat.att_sens_bias_len+self.sat.act_bias_len - 1 + self.quat_as_vec
        #     # oc[0:p,p:] = 0
        #     # oc[p:,0:p] = 0
        #     ab0 = self.sat.state_len - 1 + self.quat_as_vec
        #     ab1 = self.sat.state_len - 1 + self.quat_as_vec + self.sat.act_bias_len
        #     sb0,sb1 = ab0 + self.sat.att_sens_bias_len,ab1 + self.sat.att_sens_bias_len
        #     d0 = sb1
        #     # oc[sb0:sb1,sb0:sb1] += np.diagflat(np.sum(oc[ab0:ab1,sb0:sb1],axis = 0))
        #     # oc[sb0:sb1,sb0:sb1] += np.diagflat(np.sum(oc[d0:,sb0:sb1],axis = 0))
        #     # oc[ab0:ab1,ab0:ab1] += np.diagflat(np.sum(oc[ab0:ab1,sb0:sb1],axis = 1))
        #     # oc[d0:,d0:] += np.diagflat(np.sum(oc[d0:,sb0:sb1],axis = 1))
        #     # oc[d0:,d0:] += np.diagflat(np.sum(oc[d0:,ab0:ab1],axis = 1))
        #     # oc[ab0:ab1,ab0:ab1] += np.diagflat(np.sum(oc[d0:,ab0:ab1],axis = 0))
        #     oc[ab0:ab1,sb0:sb1] = 0
        #     oc[sb0:sb1,ab0:ab1] = 0
        #     oc[ab0:ab1,d0:] = 0
        #     oc[d0:,ab0:ab1] = 0
        #     oc[sb0:sb1,d0:] = 0
        #     oc[d0:,sb0:sb1] = 0
        # self.full_state.set_indices(self.use,out.val,oc,square_mat_sections(self.full_state.int_cov,self.cov_use()),[3]*(not self.quat_as_vec))
        # self.use_state = self.full_state.pull_indices(self.use,[3]*(not self.quat_as_vec))
        # self.sat.match_estimate(self.full_state,self.update_period)
        # self.os_vecs = os_local_vecs(os,self.full_state.val[3:7])
        # return self.full_state.val[0:self.sat.state_len]



    def predictive_step(self,os,control_vec):
        prev_control_vec = np.copy(control_vec)
        os = os.copy()

        #take out values
        state0 = self.use_state.val
        dyn_state0 = state0[0:self.sat.state_len].copy()
        cov0 = self.use_state.cov

        #find repeated orbital state once
        mid_os = self.prev_os.average(os)
        #propagate
        state1 = state0.copy()#non-base state parameters remain constant
        state1[0:self.sat.state_len] = self.sat.rk4(state0[0:self.sat.state_len],control_vec,self.update_period,self.prev_os,os,mid_orbital_state=mid_os)


        #find Jacobians
        A0,B0,dab0,dsb0,ddmp0 = self.sat.rk4Jacobians(dyn_state0,control_vec,self.update_period,self.prev_os,os,mid_orbital_state=mid_os)
        if self.quat_as_vec:
            B,combo = B0,np.vstack([A0,dab0,dsb0,ddmp0])
            combo = combo[self.use,:]
        else:
            T2 = block_diag(np.eye(3),quat_to_vec3_deriv(state1[3:7],self.vec_mode),np.eye(self.sat.state_len - 7))
            T1 =  block_diag(np.eye(3),vec3_to_quat_deriv(quat_to_vec3(state0[3:7],self.vec_mode),self.vec_mode),np.eye(self.sat.state_len - 7))
            B = B0@T2
            combo = np.vstack([T1@A0,dab0,dsb0,ddmp0])@T2
            combo_use = np.concatenate([self.use[:3],self.use[4:]])
            combo = combo[combo_use,:]

        #make big Jacobians for full state
        BigA = np.eye(np.size(self.use_state,0) - 1 + self.quat_as_vec)
        BigB = np.zeros((self.sat.control_len,np.size(self.use_state,0) - 1 + self.quat_as_vec))
        # breakpoint()
        BigA[:,:self.sat.state_len - 1 + self.quat_as_vec] = combo
        BigB[:,:self.sat.state_len - 1 + self.quat_as_vec] = B

        #covaraince propagation
        cov1 =  self.use_state.int_cov + BigA.T@cov0@BigA + BigB.T@self.sat.control_cov()@BigB
        #sensor biases go through udate even if unused, because biases drift even if you don't sample
        ##let disabled disturbances go for now, because their values and covariances should be 0.
        res =  estimated_nparray(state1,cov1)
        return res

class GaussianExtraEstimator(PerfectEstimator):
    def set_cov(self,cov):
        self.cov = cov

    def update(self, sensors_in, control_vec, os,which_sensors = None,prop_on = False,truth = None):
        err_state = np.random.multivariate_normal(truth.flatten(),self.cov).reshape((self.sat.state_len,1))
        err_state[3:7,:] = normalize(err_state[3:7,:])
        super().update(sensors_in, control_vec, os,[False for j in range(len(self.sat.sensors))],prop_on,err_state,use_truth = True,truth_cov = self.cov)
        self.state_full.val[0:self.sat.state_len,:] = np.copy(err_state)
        self.state_full.val[3:7,:] = normalize(self.state_full.val[3:7,:])
        return self.state_full.val[0:self.sat.state_len,:].reshape((self.sat.state_len,1))

class GaussianEstimator(Estimator):
    def __init__(self,cov,sat=None,sample_time = 1,estimate=None):
        self.reset(cov,sat,sample_time,estimate)

    def reset(self,cov,sat = None,sample_time=1,estimate=None):
        if sat is None:
            sat = Satellite()
            for j in range(len(sat.sensors)):
                if sat.sensors[j].has_bias:
                    jbs = sat.sensors[j].bias.size
                    if not isinstance(sat.sensors[j].bias,estimated_nparray):
                        sat.sensors[j].bias = estimated_nparray(sat.sensors[j].bias,np.diagflat(sat.sensors[j].bias_std_rate*np.ones((jbs,1))),np.diagflat(sat.sensors[j].bias_std_rate**2*np.ones((jbs,1))))

            for j in range(len(sat.act_noise)):
                if sat.act_noise[j].has_bias:
                    sat.act_noise[j].bias = estimated_float(sat.act_noise[j].bias,sat.act_noise[j].bias_std_rate,sat.act_noise[j].bias_std_rate**2)
        else:
            for j in range(len(sat.sensors)):
                if sat.sensors[j].has_bias:
                    jbs = sat.sensors[j].bias.size
                    if not isinstance(sat.sensors[j].bias,estimated_nparray):
                        sat.sensors[j].bias = estimated_nparray(sat.sensors[j].bias,np.diagflat(sat.sensors[j].bias_std_rate*np.ones((jbs,1))),np.diagflat(sat.sensors[j].bias_std_rate**2*np.ones((jbs,1))))
            for j in range(len(sat.act_noise)):
                if sat.act_noise[j].has_bias:
                    jbs = sat.act_noise[j].bias.size
                    if not isinstance(sat.act_noise[j].bias,estimated_float):
                        sat.act_noise[j].bias = estimated_float(sat.act_noise[j].bias.val,sat.act_noise[j].bias_std_rate,sat.act_noise[j].bias_std_rate**2)

        if estimate is None:
            estimate = np.zeros((sat.state_len,1))
            estimate[3:7,:] = np.array([[0.5,0.5,0.5,0.5]]).T
        self.state_full = estimated_nparray(estimate,cov,0*cov)
        self.update_period = sample_time
        self.include_gendist = False
        self.estimate_dist = False

        self.sat = sat
        self.cov = cov

        # extras_len = self.dist_inds()[-1] - self.sensor_bias_inds()[-1][1]
        # self.cross_term_main_extras = np.zeros((self.sat.state_len+self.sensor_bias_inds()[-1][1],extras_len))
        # self.extras_cov = np.zeros((extras_len,extras_len))
        self.prev_os = Orbital_State(0,np.array([[0,0,1]]).T,np.array([[0,0,0]]).T)

    def update(self, sensors_in, control_vec, os,which_sensors = None,prop_on = False,truth = None):
        # super().update(sensors_in, control_vec, os,which_sensors,prop_on,truth,use_truth = True)
        err_state = np.random.multivariate_normal(truth.flatten(),self.cov).reshape((self.sat.state_len,1))
        err_state[3:7,:] = normalize(err_state[3:7,:])
        self.state_full.val = err_state# = estimated_nparray(err_state,self.cov,0*self.cov)
        return self.state_full.val[0:self.sat.state_len,:].reshape((self.sat.state_len,1))
