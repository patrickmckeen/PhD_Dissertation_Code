from .control_mode import *

class RWBdot(ControlMode):
    def __init__(self,sat,gain_info=[1],include_disturbances=False):
        ModeName = GovernorMode.RWBDOT_WITH_EKF

        params = Params()

        params.gain = gain_info[0]

        if len(gain_info) < 2:
            params.c_gain = 0.005
        else:
            params.c_gain = gain_info[1]

        if len(gain_info) < 3:
            params.RW_bias_h = np.array([0.05*j.max_h for j in sat.actuators if isinstance(j,RW)])
        else:
            params.RW_bias_h = gain_info[2] #TODO: make bias adaptive to move more quickly through stiction, etc.
        # params.gain = gain

        super().__init__(ModeName,sat,params,True,include_disturbances,False,False)

    def find_actuation(self, state, os, osp1, goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
        """
        This function finds the commanded control input using bdot at a specific point
        in a trajectory, based on the rate of change of the magnetic field, estimated
        using the angular velocity and current magnetic field reading. Equivalent
        to control mode GovernorMode.BDOT_WITH_EKF.

        Parameters
        ------------
            db_body: np array (3 x 1)
                derivative of the magnetic field in body coordinates, in T
        Returns
        ---------
            u_out: np array (3 x 1)
                magnetic dipole to actuate for bdot, in Am^2 and body coordinates
        """
        # breakpoint()
        base_torq = -state[0:3]*self.params.gain
        q = state[3:7]
        vecs = os_local_vecs(os,q)
        return self.general_ctrl_from_torq(state,state[0:3],os,vecs,self.params.RW_bias_h,np.ones(self.rw_N)*self.params.c_gain,base_torq,np.ones(self.mtq_N)*1e0,np.ones(self.magic_N)*1e-8,np.ones(self.rw_N)*1e6,wdesdot = np.zeros(3))

        # if self.include_disturbances:
        #     moddist = self.sat.dist_torque(state,vecs)
        #     base_torq -= moddist
        #     self.saved_dist = moddist.copy()
        #
        #
        # u_rw = np.linalg.pinv(self.RWaxes)@base_torq
        # return self.RW_ctrl_matrix.T@u_rw - np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])
        # # return self.mtqrw_from_torqdes(self.bdot_gain*(-self.sat.J@w/self.update_period + cross(w,self.sat.J@w + RW_h)),RW_h,Bbody,is_fake)
