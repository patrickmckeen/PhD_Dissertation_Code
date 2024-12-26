
l[self.sat.number_MTQ:self.sat.number_MTQ+self.sat.number_RW,:] = np.array([max(l_rw[j,0],
                    -((alph*self.sat.RW_saturation[j] - RW_h[j,0])/self.update_period + self.sat.RW_J[j]*(self.sat.RW_z_axis[j].T@self.sat.invJ_noRW@(-np.cross(w,ang_mom) + torq_des )).item())/(1+self.sat.RW_J[j]*(self.sat.RW_z_axis[j].T@self.sat.invJ_noRW@self.sat.RW_z_axis[j]).item())
                    ) for j in range(self.sat.number_RW)]).reshape((self.sat.number_RW,1))
u[self.sat.number_MTQ:self.sat.number_MTQ+self.sat.number_RW,:] = np.array([min(u_rw[j,0],
                    -((-alph*self.sat.RW_saturation[j] - RW_h[j,0])/self.update_period + self.sat.RW_J[j]*(self.sat.RW_z_axis[j].T@self.sat.invJ_noRW@(-np.cross(w,ang_mom) + torq_des )).item())/(1+self.sat.RW_J[j]*(self.sat.RW_z_axis[j].T@self.sat.invJ_noRW@self.sat.RW_z_axis[j]).item())
                    ) for j in range(self.sat.number_RW)]).reshape((self.sat.number_RW,1))
