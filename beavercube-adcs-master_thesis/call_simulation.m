close all

%%%SET CONSTANTS -- change at own risk :)
%Gravitational constant G
grav_const = 6.6742e-11; %m^3*kg^−1*s^−2
%Mass of Earth
m_earth = 5.9736e+24; %kg
%Radius of Earth
r_earth = 6361010; %m

%LOAD TRAJECTORY OPTIMIZER PLACEHOLDERS
t_start = 300;
t_end = 3898;
X_traj = load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrXset.mat').Xset;
U_traj = load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrUset.mat').Uset;
B = load('../clean-rpi/beavercube-adcs/bc/matfiles/B_ECI.mat').B_ECI;
K = load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrKset_lqr.mat').Kset_lqr;
Bset = B;

%LOAD R, V, B, SUN
load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/r_ECI.mat');
load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/v_ECI.mat');
load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/sun_ECI.mat');
load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/B_ECI.mat');
load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/Xdesset_backwardspass.mat');
r_ECI = r_ECI*1000;
v_ECI = v_ECI*1000;


%SET INITIAL STATE
w_mag_mean = deg2rad(0.5);
w_mag_var = deg2rad(0.1);
w_bias_mean = 0;
w_bias_var = deg2rad(5);
x_init = [0.000653456389222607 0.00213372325947466 0.00798525589576257 0.539902920364321 -0.517355822225068 0.661750161605261 0.0541711492153578].';%[generate_w(w_mag_mean, w_mag_var); generate_q()]; %w in rad/s%[0.0067 0.0067 0.0046 -0.1096 -0.5967 0.5336 0.5893]';%[generate_q(); generate_w(w_mag_mean, w_mag_var)]; %w in rad/s
w_bias_init = [-0.0116 -0.0624 0.1179]';
torque_init = zeros(3, 1);%cross(rotT(X_traj(4:7, 1))*B(301, :).', U_traj(:, 1));%[-5.55105598061822e-06; -5.4967188687378e-06; -8.30223718471259e-08];


%DIFFERENT SIMULINK BLOCKS
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

%EKF
ekf_initial_guess = [0.5; 0.5; 0.5; 0.5; 0; 0; 0; 0; 0; 0]; %q (unitless), w (rad/s), w bias (rad/s)
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

%RUN SIMULATION
set_param('bcadcs','StartTime','1','StopTime','3598')
simOut2 = sim('bcadcs.slx');


%DATA PROCESSING
x_out_raw = simOut2.yout{1}.Values.Data;
u_out_raw = simOut2.yout{2}.Values.Data;
m_out_raw = simOut2.yout{3}.Values.Data;
m_bounded_raw = simOut2.yout{4}.Values.Data;
B_out_raw = simOut2.yout{5}.Values.Data;
xin_raw = simOut2.yout{6}.Values.Data;
xout_raw = simOut2.yout{7}.Values.Data;
xpredict_raw = simOut2.yout{13}.Values.Data;
xerr_raw = simOut2.yout{14}.Values.Data;
gyrobiaspredict_raw = simOut2.yout{15}.Values.Data;
gyrobiastrue_raw = simOut2.yout{16}.Values.Data;
xpredict_untrunc_raw = simOut2.yout{17}.Values.Data;

x_rawsize = size(x_out_raw);
x_out = zeros(x_rawsize(1), x_rawsize(3));
for ind=1:x_rawsize(3)
    x_out(:, ind) = x_out_raw(:, 1, ind);
end
%x_out = x_out_raw.';

xpredict_rawsize = size(xpredict_raw);
xpredict = zeros(xpredict_rawsize(1), xpredict_rawsize(3));
for ind=1:xpredict_rawsize(3)
    xpredict(:, ind) = xpredict_raw(:, 1, ind);
end

gyrobiaspredict_rawsize = size(gyrobiaspredict_raw);
gyrobiaspredict = zeros(gyrobiaspredict_rawsize(1), gyrobiaspredict_rawsize(3));
for ind=1:gyrobiaspredict_rawsize(3)
    gyrobiaspredict(:, ind) = gyrobiaspredict_raw(:, 1, ind);
end

%FIGURES

figure()
hold on
plot(m_bounded_raw(300:400,1))
plot(U_traj(1,1:100).')
legend('actual dipole', 'trajectory planner dipole');
xlabel('Time elapsed (s)');
ylabel('Dipole (Am^2)');
hold off



figure()
hold on
plot(x_out(:, 300:end).')
plot(X_traj(:, 1:3598).')
hold off

% figure()
% hold on
% plot(m_out_raw)
% hold off
% 
% figure()
% hold on
% plot((x_out(4, 1:3598)-xpredict(4, 1:3598)).');
% plot((x_out(5, 1:3598)-xpredict(5, 1:3598)).');
% plot((x_out(6, 1:3598)-xpredict(6, 1:3598)).');
% plot((x_out(7, 1:3598)-xpredict(7, 1:3598)).');
% plot((xpredict(4:7, 1:3598)).');%;


% hold off
% xlabel('Time elapsed (s)');
% ylabel('Quaternion');
% title('Actual and EKF-predicted quaternion');
% legend('q1 diff', 'q2 diff', 'q3 diff', 'q4 diff');
% 
% figure()
% hold on
% plot((x_out(1, 20:3598)-xpredict(1, 20:3598)).');%;
% plot((x_out(2, 20:3598)-xpredict(2, 20:3598)).');%;
% plot((x_out(3, 20:3598)-xpredict(3, 20:3598)).');%;
% hold off
% xlabel('Time elapsed (s)');
% ylabel('Angular velocity (rad/s)');
% title('Actual and EKF-predicted angular velocity (rad/s)');
% legend('w1 diff', 'w2 diff', 'w3 diff');
% 
% 
% figure()
% hold on
% plot(xerr_raw(1:100, :));
% hold off
% xlabel('Time elapsed (s)');
% ylabel('X error in controller');
% title('X error at start of simulation');
% legend('wx', 'wy', 'wz', 'quat1', 'quat2', 'quat3');

% figure()
% plot(x_err_raw)
% 
% figure()
% plot(x_err_raw(1:10, :))
ekf_convergence_error = zeros(300, 1);
for ind = 1:300
    ekf_convergence_error(ind) = quat_distance(x_out(4:7, ind), xpredict(4:7, ind));
end
figure()
plot(ekf_convergence_error);
xlabel('Time elapsed (s)');
ylabel('Angle between EKF estimate and true quaternion (deg)');
title('EKF Convergence Error Over Time');

ekf_controlon_error = zeros(3298, 1);
for ind=300:3598
    ekf_controlon_error(ind-299) = quat_distance(x_out(4:7, ind), xpredict(4:7, ind));
end
figure()
hold on
plot(ekf_controlon_error);
xlabel('Time elapsed (s)');
ylabel('Angle between EKF estimate and true quaternion (deg)');
title('EKF Error Over Time: Controller On & Using True State');
%GET MEAN AND STDEV
controlon_std = std(ekf_controlon_error);
controlon_mean = mean(ekf_controlon_error);
%ANNOTATE
txt = 'Mean Error: ' + string(controlon_mean) + newline + 'Standard Deviation: ' + string(controlon_std);
dim = [.6 .5 .3 .3];
annotation('textbox',dim,'String',txt,'FitBoxToText','on');
hold off

ekf_control_error = zeros(3298, 1);
for ind=300:3598
    ekf_control_error(ind-299) = quat_distance(x_out(4:7, ind), X_traj(4:7, ind-299));
end
figure()
hold on
plot(((ekf_control_error)));
plot(((ekf_controlon_error)));
legend('Angle between desired and true quaternion', 'Angle between desired and estimated quaternion');
xlabel('Time elapsed (s)');
ylabel('Error Angle (deg)');
title('Control Error (deviation from optimal trajectory)');
% %GET MEAN AND STDEV
% controlon_std = std(ekf_control_error);
% controlon_mean = mean(ekf_control_error);
% %ANNOTATE
% txt = 'Mean Error: ' + string(controlon_mean) + newline + 'Standard Deviation: ' + string(controlon_std);
% dim = [.6 .5 .3 .3];
% annotation('textbox',dim,'String',txt,'FitBoxToText','on');
hold off

%PLOT QUATERNION AND ESTIMATED QUATERNION
% figure()
% hold on
% plot(x_out(4:7, 300:end).')
% plot(xpredict(4:7, 300:end).')
% hold off
angvel_estimation_error = zeros(3598, 1);
angvel_estimation_mag = zeros(3598, 1);
for ind=1:3598
    [ang, mag] = angvel_distance(x_out(1:3, ind), xpredict(1:3, ind));
    angvel_estimation_error(ind) = ang;
    angvel_estimation_mag(ind) = mag;
end
figure()
plot(angvel_estimation_error(1:300));
title('EKF Convergence Error: Angular Velocity Estimate Over Time');
ylabel('Angle between true and estimated angular velocity (deg)');
xlabel('Time elapsed (s)');

figure()
hold on
plot(angvel_estimation_error(300:end));
title('EKF Steady-State Error: Angular Velocity Estimate Over Time');
ylabel('Angle between true and estimated angular velocity (deg)');
xlabel('Time elapsed (s)');
%GET MEAN AND STDEV
controlon_std = std(angvel_estimation_error(300:end));
controlon_mean = mean(angvel_estimation_error(300:end));
%ANNOTATE
txt = 'Mean Error: ' + string(controlon_mean) + newline + 'Standard Deviation: ' + string(controlon_std);
dim = [.6 .5 .3 .3];
annotation('textbox',dim,'String',txt,'FitBoxToText','on');
hold off

figure()
hold on
plot(angvel_estimation_mag(300:end));
title('EKF Steady-State Error: Angular Velocity Estimate Over Time');
ylabel('Ratio of Estimate to True Angular Velocity');
xlabel('Time elapsed (s)');
%GET MEAN AND STDEV
controlon_std = std(angvel_estimation_mag(300:end));
controlon_mean = mean(angvel_estimation_mag(300:end));
%ANNOTATE
txt = 'Mean: ' + string(controlon_mean) + newline + 'Standard Deviation: ' + string(controlon_std);
dim = [.6 .5 .3 .3];
annotation('textbox',dim,'String',txt,'FitBoxToText','on');
hold off

gyrobias_estimation_error = zeros(3598, 1);
gyrobias_estimation_mag = zeros(3598, 1);
for ind=1:3598
    [ang, mag] = angvel_distance(gyrobiastrue_raw(ind, 1:3).', gyrobiaspredict(1:3, ind));
    gyrobias_estimation_error(ind) = ang;
    gyrobias_estimation_mag(ind) = mag;
end

figure()
hold on
plot(gyrobias_estimation_error(300:end));
title('EKF Steady-State Error: Gyroscope Bias Estimate Over Time');
ylabel('Angle between true and estimated gyro bias (deg)');
xlabel('Time elapsed (s)');
%GET MEAN AND STDEV
controlon_std = std(gyrobias_estimation_error(300:end));
controlon_mean = mean(gyrobias_estimation_error(300:end));
%ANNOTATE
txt = 'Mean Error: ' + string(controlon_mean) + newline + 'Standard Deviation: ' + string(controlon_std);
dim = [.6 .5 .3 .3];
annotation('textbox',dim,'String',txt,'FitBoxToText','on');
hold off

figure()
hold on
plot(gyrobias_estimation_mag(300:end));
title('EKF Steady-State Error: Gyroscope Bias Estimate Over Time');
ylabel('Ratio of Estimate to True Gyro Bias');
xlabel('Time elapsed (s)');
%GET MEAN AND STDEV
controlon_std = std(gyrobias_estimation_mag(300:end));
controlon_mean = mean(gyrobias_estimation_mag(300:end));
%ANNOTATE
txt = 'Mean: ' + string(controlon_mean) + newline + 'Standard Deviation: ' + string(controlon_std);
dim = [.6 .5 .3 .3];
annotation('textbox',dim,'String',txt,'FitBoxToText','on');
hold off

%Functions
function Rt = rotT(q) %ECI->body, vbody = R*vECI
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    q4 = q(4);
    
    Rt = [q1^2+q2^2-q3^2-q4^2 2*(q2*q3-q1*q4) 2*(q2*q4+q1*q3);...
        2*(q2*q3+q1*q4) q1^2-q2^2+q3^2-q4^2 2*(q3*q4-q1*q2);...
        2*(q2*q4-q1*q3) 2*(q3*q4+q1*q2) q1^2-q2^2-q3^2+q4^2].';
end
function theta = quat_distance(q1, q2)
    theta = acosd(2*dot(q1, q2)^2 - 1);
end
function [theta, mag] = angvel_distance(v1, v2)
    if norm(v1) ~= 0 && norm(v2) ~= 0
        theta = acosd(dot(v1, v2)/(norm(v1)*norm(v2)));
        mag = rad2deg(norm(v2)-norm(v1));
    elseif norm(v1) == 0 && norm(v2) == 0
        theta = 0;
        mag = 1;
    else
        theta = 180;
        mag = max(norm(v2), norm(v1));
    end
    
end
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