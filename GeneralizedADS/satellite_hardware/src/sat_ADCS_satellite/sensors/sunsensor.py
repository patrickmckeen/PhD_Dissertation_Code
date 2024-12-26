from .sensor import *

class SunSensor(Sensor):
    """
    This class extends the Sensor class for a SINGLE coarse sun sensor.
    """
    #TODO make solar constant vary

    def __init__(self, axis,std, efficiency, sample_time = 0.1, has_bias = True, bias = None, bias_std_rate = None, use_noise = True,degradation_value = 0,estimate_bias = False,scale = 1,respect_eclipse = True):
        """
        Initialize the sun sensors.
        See class definition.

        Already area-normalized and normalized to solar_constant
        """
        if bias_std_rate is None:
            bias_std_rate = np.zeros(1)
        if bias is None:
            bias = np.zeros(1)
        noise_model = lambda val,noise_settings,state,vecs: np.maximum(0,np.random.normal(val,noise_settings[0])) #cannot return negative
        noise_update_model = lambda val,noise_settings,state,vecs,dt: noise_settings
        self.axis = normalize(axis)
        self.std = std
        self.efficiency = efficiency
        self.respect_eclipse = respect_eclipse
        self.degradation_value = degradation_value #this is the YD factor in solar panel decay calculations, but must be adjusted for continous degradation vs yearly compounding
        super().__init__(1, sample_time, has_bias, bias,bias_std_rate,use_noise,noise_model,noise_update_model,(self.std,degradation_value,np.nan),estimate_bias,scale)

    def clean_reading(self,state,vecs):
        return self.efficiency*np.maximum(np.dot(normalize(vecs['s']-vecs['r']),self.axis),0)*self.scale*(not (vecs["os"].in_eclipse() and self.respect_eclipse))

    def bias_jac(self,x,vecs):
        if self.has_bias:
            return np.ones((1,1))
        else:
            return np.zeros((0,1))

    def basestate_jac(self,x,vecs):
        sunvec = vecs['s']-vecs['r']
        ns = normalize(sunvec)
        dns__dq = normed_vec_jac(sunvec,vecs["ds"]-vecs["dr"])#(dsb__dq-drb__dq)#/norm(S_B-R_B)
        cos_incidence = np.dot(ns,self.axis)
        dcos_incidence__dq = (cos_incidence>0)*(dns__dq@self.axis)
        return np.vstack([np.zeros((3,1)),self.efficiency*np.expand_dims(dcos_incidence__dq,1)])*self.scale*(not (vecs["os"].in_eclipse() and self.respect_eclipse))

    def update_noise(self,biased_val,state,vecs):
        if self.use_noise:
            new_settings = self.noise_update_model(biased_val/self.scale,self.noise_settings,state,vecs)
            if np.isnan(new_settings[2]):
                new_settings[2] = vecs['os'].J2000
            else:
                dt =  vecs['os'].J2000 - new_settings[2]
                self.efficiency *= math.exp(-100*dt*new_settings[1])
            self.noise_settings = new_settings