%%%SET CONSTANTS -- change at own risk :)
%Gravitational constant G
grav_const = 6.6742e-11; %m^3*kg^−1*s^−2
%Mass of Earth
m_earth = 5.9736e+24; %kg
%Radius of Earth
r_earth = 6361010; %m

%%%SET PARAMETERS
%Simulation length
sim_length = 5520; %~1 ISS orbit

%Magnetic field, orbital position, orbital velocity, and sun position
%Run this to recalculate r_ECI, B_ECI, and v_ECI for your slew length
%(warning: slow)
% r_init = 10^3*[-4.0460; 5.4515; 0.2396]; %km
% v_init = [-3.6945; -3.0058; 6.0022]; %km/s
% [r_ECI, B_ECI, v_ECI] = predict_Br(r_init, v_init, 1, sim_length, 1, 1);
% r_ECI = r_ECI*10^3; %m
% v_ECI = v_ECI*10^3; %m
% save('matfiles/r_ECI.mat', 'r_ECI'); 
% save('matfiles/v_ECI.mat', 'v_ECI');
% save('matfiles/B_ECI.mat', 'B_ECI');
%Otherwise, just load these values
load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/r_ECI.mat');
load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/v_ECI.mat');
load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/sun_ECI.mat');
load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/B_ECI.mat');
load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/Xdesset_backwardspass.mat');
r_ECI = r_ECI*1000;
v_ECI = v_ECI*1000;
% load('matfiles/B_ECI_equatorial.mat');
% load('matfiles/sun_ECI.mat');
%r_ECI = r_ECI_equatorial;
%v_ECI = v_ECI_equatorial;
% B_ECI = B_ECI_equatorial;
%load('matfiles/magfield_ECI.mat')
%B_ECI = magfield_ECI;

%Initial state -- this should NOT be random if you intend on using LQR
%control
w_mag_mean = deg2rad(0.5);
w_mag_var = deg2rad(0.1);
w_bias_mean = 0;
w_bias_var = deg2rad(5);
x_init = [0.000653456389222607 0.00213372325947466 0.00798525589576257 0.539902920364321 -0.517355822225068 0.661750161605261 0.0541711492153578].';%[generate_w(w_mag_mean, w_mag_var); generate_q()]; %w in rad/s%[0.0067 0.0067 0.0046 -0.1096 -0.5967 0.5336 0.5893]';%[generate_q(); generate_w(w_mag_mean, w_mag_var)]; %w in rad/s
w_bias_init = [-0.0116 -0.0624 0.1179]'; %generate_wbias(w_bias_mean, w_bias_var); %rad/s

%Dynamics
J = [31364908.06, -6713613.57, 58830.40;
    -6713613.57,10040919.97,-123347.56;
    58830.40,-123347.56,34091278.27]*10^-9; %kg*m^2

%Sensors
b_noise_var = (10^-6)^2; %nT
w_noise_var = (8.2788e-04)^2;  %rad/s
w_bias_noise_var = (2.4241e-05)^2; %rad/s
s_noise_var = (0.01)^2; %percentage of max I

%Actuators
m_noise_var = 0; %Am^2

%Disturbance torques
dipole_var = 0.05;
com_var_percentage = 0.1;
Cd = 2.2;
prop_mag = 8.5*10^-7; %Nm
spacecraft_COM = generate_com(com_var_percentage);
prop_axis = cross(([0; 0.034; -0.15]-spacecraft_COM), [0, 0, -1]);
prop_axis = prop_axis/norm(prop_axis);
residual_dipole = generate_residual_dipole(dipole_var); %Am^2
SMAD_altrange = [0 100 150 175 200 225 250 275 300 325 350 375 400 450 500 550 600 650 700 750 800 850 900 950 1000]; %km
SMAD_airdensity_vsalt = [1.2; 5.69*10^-7; 2.02*10^-9; 7.66*10^-10; 2.90*10^-10; 1.46*10^-10; 7.30*10^-11; 4.10*10^-11; 2.30*10^-11; 1.38*10^-11; 8.33*10^-12; 5.24*10^-12; 3.29*10^-12; 1.39*10^-12; 6.15*10^-13; 2.84*10^-13; 1.37*10^-13; 6.87*10^-14; 3.63*10^-14; 2.02*10^-14; 1.21*10^-14; 7.69*10^-15; 5.24*10^-15; 3.78*10^-15; 2.86*10^-15]; %kg/m^3
%Status of different torques (on = 1)
gravitygradient_on = 0;
aero_on = 0;
srp_on = 0;
dipole_on = 0;
prop_on = 0;

%Controller
controller_status = 2; %0 for off, 1 for bdot, 2 for LQR
X_init = zeros(7, 3599);
U_init = zeros(3, 3598);
K_init = zeros(18, 3598);
 N = 3599;
J = [31364908.06, -6713613.57, 58830.40;
    -6713613.57,10040919.97,-123347.56;
    58830.40,-123347.56,34091278.27]*10^-9;
qtest2 = [0.217460776886359;         0.208932903282972;         0.852787360338662;        -0.426393680169331];
x0 = [-0.0002; 0.0005; 0.0001; qtest2/norm(qtest2)];


dt = 1.0;
rho = 0.0;
drho = 0.0;
mu = 40.0;
regMin = 10^-8;
regScale = 1.6;
Nslew = 0;
rNslew = [-3635473.26033436; 5715151.98393498; 0.0];
swslew = 0.0001;%0.0001;%10/max(abs(wdes)+0.01,[],'all')^2; %importance of angular velocity during slew or off-freq
swpoint = 1e2*(rad2deg(1))^2;%100/max(abs(wdes)+0.01,[],'all')^2;%importance of angular velocity during regular times
sv1 = 1e4;%1e8; %importance of aoreitnation or angular difference
sv2 = 1e3;
su = 1e3;%1/max(abs(mdes)+1e-6,[],'all'); %importance of u
sratioslew = 0.001;
R = eye(3)*1000;
QN = [eye(3)*1e4 zeros(3,3);
      zeros(3,3) eye(3)*328280.635001174];
 
 
maxIlqrIter = 35;
beta1 = 10^-8;
beta2 = 5;
regBump = 1e8;
xmax = ones(7, 1)*10;
umax = ones(3, 1)*0.15;
eps = 2.2204*10^-16;
regInit = 0;
maxLsIter = 10;
gradTol = 1e-6;
costTol = 0.001;%10^-4;
%zCountLim = 10;
zCountLim=1;
maxIter = 700;
ilqrCostTol = 8.0000e-05;

cmax = 1e-3;
slackmax = 1e-8;
maxOuterIter = 15;
dJcountLim = 10;
penMax = 1e18;%1e8;
penScale = 50;
penInit = 100;
lagMultMax = 1e10;
lagMultInit = 0;

save('../clean-rpi/beavercube-adcs/bc/matfiles/trajOptSettings.mat', 'N', 'J', 'x0', 'dt', 'rho', 'drho', 'mu', 'regMin', 'regScale', 'Nslew', 'sv1', 'swpoint', 'rNslew', 'swslew', 'sratioslew', 'R', 'QN', 'maxLsIter', 'beta1', 'beta2', 'regBump', 'xmax', 'umax', 'eps', 'lagMultInit', 'penInit', 'regInit', 'maxOuterIter',...
    'maxIlqrIter', 'gradTol', 'costTol', 'cmax', 'zCountLim', 'maxIter', 'penMax', 'penScale', 'lagMultMax', 'ilqrCostTol');

% dt = 1.0;
% rho = 0.0;
% drho = 0.0;
% mu = 40.0;
% regMin = 10^-8;
% regScale = 1.6;
% Nslew = 0;
% rNslew = [-3635473.26033436; 5715151.98393498; 0.0];
% swslew = 0.0001;%0.0001;%10/max(abs(wdes)+0.01,[],'all')^2; %importance of angular velocity during slew or off-freq
% swpoint = 1e2*(rad2deg(1))^2;%100/max(abs(wdes)+0.01,[],'all')^2;%importance of angular velocity during regular times
% sv1 = 1e4;%1e8; %importance of aoreitnation or angular difference
% sv2 = 1e3;
% su = 1e3;%1/max(abs(mdes)+1e-6,[],'all'); %importance of u
% sratioslew = 0.001;
% R = eye(3)*1000;
% QN = [eye(3)*1e4 zeros(3,3);
%       zeros(3,3) eye(3)*328280.635001174];
%  
%  
% maxIlqrIter = 45;
% beta1 = 10^-8;
% beta2 = 5;
% regBump = 1e7;
% xmax = ones(7, 1)*10;
% umax = ones(3, 1)*0.15;
% eps = 2.2204*10^-16;
% regInit = 0;
% maxLsIter = 10;
% gradTol = 1e-6;
% costTol = 10^-4;
% zCountLim = 10;
% maxIter = 700;
% ilqrCostTol = 8.0000e-05;
% 
% cmax = 1e-3;
% slackmax = 1e-8;
% maxOuterIter = 15;
% dJcountLim = 10;
% penMax = 1e18;%1e8;
% penScale = 50;
% penInit = 100;
% lagMultMax = 1e10;
% lagMultInit = 0;
% 
% save('matfiles/trajOptSettings.mat', 'N', 'J', 'x0', 'dt', 'rho', 'drho', 'mu', 'regMin', 'regScale', 'Nslew', 'sv1', 'swpoint', 'rNslew', 'swslew', 'sratioslew', 'R', 'QN', 'maxLsIter', 'beta1', 'beta2', 'regBump', 'xmax', 'umax', 'eps', 'lagMultInit', 'penInit', 'regInit', 'maxOuterIter',...
%     'maxIlqrIter', 'gradTol', 'costTol', 'cmax', 'zCountLim', 'maxIter', 'penMax', 'penScale', 'lagMultMax', 'ilqrCostTol');


%EKF
ekf_initial_guess = [0.5; 0.5; 0.5; 0.5; 0; 0; 0; 0.001; 0.001; 0.001]; %q (unitless), w (rad/s), w bias (rad/s)
Vk = eye(3)*0.01;
Qk = [5.3924e-04 zeros(1, 8);
     0  1.2210e-04 zeros(1, 7);
     zeros(1, 2) 0.0013 zeros(1, 6);
     zeros(1, 3) 1.9186e-05 zeros(1, 5);
     zeros(1, 4) 2.5486e-05 zeros(1, 4);
     zeros(1, 5) 2.4370e-05 zeros(1, 3);
     zeros(1, 6) 1.9460e-05 zeros(1, 2);
     zeros(1, 7) 2.5474e-05 0;
     zeros(1, 8) 2.4364e-05];
b_noise_var_guess = b_noise_var;%(5.0000e-07)^2; %nT
s_noise_var_guess = s_noise_var;%(1.0000e-02)^2; % percent of max I
w_noise_var_guess = w_noise_var+w_bias_noise_var;%(1.0000e-07)^2; % rad/s

%simOut = sim('controller_testbed.slx');


%Function to generate residual dipole
function dipole = generate_residual_dipole(dipole_var)
    dipole = [normrnd(0, dipole_var); normrnd(0, dipole_var); normrnd(0, dipole_var)];
end
%Function to generate com
function com = generate_com(com_var_percentage)
    com = zeros(3, 1);
    com(1) = normrnd(0, com_var_percentage*0.1);
    com(2) = normrnd(0, com_var_percentage*0.1);
    com(3) = normrnd(0, com_var_percentage*0.3);
end
%Functions to generate q, w, bias
function q_random  = generate_q()
    q_random = zeros(4, 1);
    q_random(1) = rand()-0.5;
    q_random(2) = rand()-0.5;
    q_random(3) = rand()-0.5;
    q_random(4) = rand()-0.5;
    q_random = q_random/norm(q_random);
end
function w_random = generate_w(w_mag_mean, w_mag_var)
    w_random = zeros(3, 1);
    w_random(1) = rand();
    w_random(2) = rand();
    w_random(3) = rand();
    w_random_mag = normrnd(w_mag_mean, w_mag_var);
    w_random = w_random/norm(w_random);
    w_random = w_random * w_random_mag;
end
function wbias_random = generate_wbias(w_bias_mean, w_bias_var)
    wbias_random = zeros(3, 1);
    wbias_random(1) = normrnd(w_bias_mean, w_bias_var);
    wbias_random(2) = normrnd(w_bias_mean, w_bias_var);
    wbias_random(3) = normrnd(w_bias_mean, w_bias_var);
end
%Functions for orbit prop and IGRF
function [r_predict, B_predict, v_predict] = predict_Br(r_init, v_init, dt, slew_length, t_start, verbose)
    B_predict = zeros(slew_length/dt, 3);
    v_predict = zeros(slew_length/dt, 3);
    r_predict = zeros(slew_length/dt, 3);
    for i = 1:dt:slew_length
        r_prev = r_init;
        v_prev = v_init;
        if i > 1 && mod(i, 50) == 0 && verbose == 1
            fprintf('%f %s of the way done calculating B, r, and v! \n', i*100/slew_length, '%');
            r_prev = r_predict(i-1,:);
            B_prev = B_predict(i-1,:);
            v_prev = v_predict(i-1,:);
        end
        mu = 3.986004418*10^5; %km^3s^-2
        r_norm = norm(r_prev);
        r_predict(i,:) = r_prev + v_prev*dt;
        v_predict(i,:) = v_prev - mu/(r_norm^3)*r_prev*dt;
        BECI = igrf_estimate_test(r_predict(i,:)*10^3, t_start+i);
        B_predict(i,:) = BECI;
    end
    B_predict = B_predict*10^-9;
end
function [B_ECI] = igrf_estimate_test(rECI, t)
    utc_time = get_utc(t);
    load('matfiles/g_coefs.mat');
    load('matfiles/h_coefs.mat');
    r_LLA = eci2lla(rECI, utc_time);
    lat = r_LLA(1); %deg
    long = r_LLA(2); %deg
    alt = r_LLA(3); %m above sea level
    colatitude = (90-lat);
    east_long = long;
    if east_long < 0
        east_long = east_long + 360;
    end
    east_long = (east_long);
    a = 6371.2*10^3;
    alt = alt+a;
    Br = 0;
    Btheta = 0;
    Bphi = 0;
    dP = zeros(14,14);
    for n=1:13
        for m=0:n
            P_n_minus_1 = legendre(n-1, cosd(colatitude), 'sch');
            if n==m && n==1
                dP(n+1, m+1) = cosd(colatitude);
            elseif n==m
                P_n_minus_n_minus = P_n_minus_1(n);
                dP_nn = sind(colatitude)*dP(n, n) + cosd(colatitude)*P_n_minus_n_minus;
                dP(n+1, m+1) = sqrt(1 - 1/(2*n))*dP_nn;
            else
                dP_nn = (2*n - 1)/sqrt(n^2 - m^2)*(cosd(colatitude)*dP(n, m+1) - sind(colatitude)*P_n_minus_1(m+1));
                K_nm = 0;
                if n > 1
                    K_nm = sqrt(((n-1)^2 - m^2)/(n^2 - m^2));
                    dP_nn = dP_nn - K_nm*dP(n-1, m+1);
                end
                dP(n+1, m+1) = dP_nn;
            end
        end
    end
    for n=1:13
        P = legendre(n, cosd(colatitude), 'sch');
        Br_n = 0;
        Btheta_n = 0;
        Bphi_n = 0;
        for m=0:n
            g_nm = g(n, m+1);
            h_nm = h(n, m+1);
            P_nm = P(m+1);
            dP_nm = dP(n+1, m+1);
            Br_nm = (g_nm*cosd(m*east_long)+h_nm*sind(m*east_long))*P_nm;
            Br_n = Br_n + Br_nm;
            Btheta_nm = (g_nm*cosd(m*east_long)+h_nm*sind(m*east_long))*dP_nm;
            Btheta_n = Btheta_n + Btheta_nm;
            Bphi_nm = m*(-g_nm*sind(m*east_long)+h_nm*cosd(m*east_long))*P_nm;
            Bphi_n = Bphi_n + Bphi_nm;
        end
        Br_n = Br_n*((a/alt)^(n+2))*(n+1);
        Br = Br + Br_n;
        Btheta_n = Btheta_n*((a/alt)^(n+2));
        Btheta = Btheta + Btheta_n;
        Bphi_n = Bphi_n*((a/alt)^(n+2));
        Bphi = Bphi + Bphi_n;
    end
    Bphi = Bphi*-1/(sind(colatitude));
    Br;
    Btheta = Btheta*-1;
    Bn = -Btheta;
    Be = Bphi;
    Bd = -Br;
    B_ned = [Bn, Be, Bd];
    [b_ecefx, b_ecefy, b_ecefz] = ned2ecefv(Bn, Be, Bd, lat, long);
    B_ECI = ecef2eci(utc_time, [b_ecefx b_ecefy b_ecefz]);
end
function utc_time=get_utc(ind)
    minutes_elapsed = 1;
    hours_elapsed = 1;
    seconds_elapsed = ind;
    if seconds_elapsed > 60
        minutes_elapsed = minutes_elapsed + floor((seconds_elapsed-1)/60);
        seconds_elapsed = seconds_elapsed - ((minutes_elapsed-1)*60);
    end
    if minutes_elapsed > 60
        hours_elapsed = hours_elapsed + floor((minutes_elapsed-1)/60);
        minutes_elapsed = minutes_elapsed - ((hours_elapsed-1)*60);
    end
    utc_time = [2020 11 24 hours_elapsed minutes_elapsed seconds_elapsed];
end