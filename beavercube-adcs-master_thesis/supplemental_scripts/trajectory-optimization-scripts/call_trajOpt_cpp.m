clear all
load('matfiles/qf_desired.mat');
t_start = 300;
N = 3899;
J = [31364908.06, -6713613.57, 58830.40;
    -6713613.57,10040919.97,-123347.56;
    58830.40,-123347.56,34091278.27]*10^-9;
qtest2 = [0.217460776886359;         0.208932903282972;         0.852787360338662;        -0.426393680169331];
x0 = [-0.0002; 0.0005; 0.0001; qtest2/norm(qtest2)];
x0 = [0.0067 0.0067 0.0046 -0.1096 -0.5967 0.5336 0.5893]';
%x0 = [generate_w(w_mag_mean, w_mag_var); generate_q()];


% x0 = [0.001;
%     0.001;
%     -0.001;
%    0.5;
%    0.5;
%     0.5;
%     0.5];
dt = 1.0;
rho = 0.0;
drho = 0.0;
mu = 40.0;
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
  
QN = [eye(3)*328280.635001174 zeros(3,3);
     zeros(3,3) eye(3)*1e4];
 
 
xmax = ones(7, 1)*10;
umax = ones(3, 1)*0.15;
eps = 2.2204*10^-16;
%zCountLim = 10;
costTol = 1e-4;
maxIter = 700;

gradTol = 1e-5;
ilqrCostTol = 10*costTol;
maxIlqrIter = 30;
beta1 = 1e-8;%1e-8;
beta2 = 10;
maxLsIter = 10;
regInit = 0;
regMax = 1e8;
regMin = 1e-8;
regBump = 1e8;
regScale = 1.6;%2;%5;%1.6;
costMax = 1e8;
zCountLim = 10;
maxControl = 1e8; %this is NOT the constraint value. This is a number to throw an error if we see it
maxState = 1e8;

cmax = 1e-3;
slackmax = 1e-8;
maxOuterIter = 15;
dJcountLim = 10;
penMax = 1e18;%1e8;
penScale = 50;
penInit = 100;
lagMultMax = 1e8;
%lagMultMin = -1e4;%-1e8;
lagMultInit = 0;

save('../clean-rpi/beavercube-adcs/bc/matfiles/trajOptSettings.mat', 'N', 'J', 'x0', 'dt', 'rho', 'drho', 'mu', 'regMin', 'regScale', 'Nslew', 'sv1', 'swpoint', 'rNslew', 'swslew', 'sratioslew', 'R', 'QN', 'maxLsIter', 'beta1', 'beta2', 'regBump', 'xmax', 'umax', 'eps', 'lagMultInit', 'penInit', 'regInit', 'maxOuterIter',...
    'maxIlqrIter', 'gradTol', 'costTol', 'cmax', 'zCountLim', 'maxIter', 'penMax', 'penScale', 'lagMultMax', 'ilqrCostTol');
Xdes = zeros(7, 5520);
for ind=1:5520
    Xdes(1:3, ind) = zeros(3, 1);
    Xdes(4:7, ind) = qf_desired(ind,:).';
    negate =0;
    if ind > 1
        for j = 4:7
            if abs(Xdes(j, ind) - Xdes(j, ind-1)) > 0.1
                negate = 1;
            end
        end
    end
    if negate==1
        Xdes(4:7, ind) = -1*qf_desired(ind,:).';
    end
end
Xdesset_backwardspass = Xdes;
save('../clean-rpi/beavercube-adcs/bc/matfiles/Xdesset_backwardspass.mat', 'Xdesset_backwardspass');
system_string = './../clean-rpi/beavercube-adcs/bc/TrajectoryPlanning/MatrixTest/build/ALTRO '+ string(N) + ' ' + string(t_start) + ' ' + string(x0(1)) + ' ' + string(x0(2)) + ' ' + string(x0(3)) + ' ' + string(x0(4)) + ' ' + string(x0(5)) + ' ' + string(x0(6)) + ' ' + string(x0(7));
system(system_string);
load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrXtraj.mat');
load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrXset.mat');
class(Xset)
X = Xset;
Uinit = load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrUinit.mat');

load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrUset.mat');
U = Uset;
load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrlambdaSet.mat');
load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrKset.mat');
K = zeros(3, 6, N);
for ind=1:N
    K(:, :, ind) = reshape(Kset(:, ind), 3, 6);
end
%load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrPset.mat');
% P = zeros(6, 6, N);
% for ind=1:N
%     P(:, :, ind) = reshape(Pset(:, ind), 6, 6);
% end
load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrMu.mat');
load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrKset_lqr.mat');
K_lqr = zeros(3, 6, N);
for ind=1:N
    K_lqr(:, :, ind) = reshape(Kset_lqr(:, ind), 3, 6);
end
close all
% figure()
% plot(Uinit.U.');
% figure()
% plot(Xtraj.');
% figure()
% plot(Uset.');
% figure()
% plot(Xset.');
load("/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/v_ECI.mat");
size_Xset = size(Xset);
length_traj = size_Xset(2);
norm_vECI = zeros(length_traj, 3);
for ind=1:length_traj
    v_ind = v_ECI(t_start+ind, :);
    norm_vECI(ind, :) = -v_ind/norm(v_ind);
end

satAlignVector1 = [0; 0; -1];
prop_point = quatrotate(quatconj(Xset(4:7,:).'),satAlignVector1.');
figure()
hold on
plot(prop_point);
plot(norm_vECI);
legend('Ux', 'Uy', 'Uz', '-Vx', '-Vy', '-Vz');
xlabel('Time elapsed (sec)');
ylabel('Position in ECI (normalized)');
title('Alignment of Propulsion System (U) with Anti-Velocity direction (-V), in ECI coordinates');
hold off
figure()
plot(rad2deg(Xset(1:3,:).'))
xlabel('Time elapsed (sec)');
ylabel('Angular velocity (deg/s)');
title('Angular Velocity over Time');

angle_error = zeros(length_traj);
for ind=1:length_traj
    angle_error(ind) = acosd(dot(prop_point(ind, :), norm_vECI(ind, :)));
end
figure()
plot(angle_error)
xlabel('Time elapsed (sec)');
ylabel('Angle error (deg)');
title('Angle between Propulsion System and Anti-Velocity direction, in degrees');

figure()
plot(Uset.');
xlabel('Time elapsed (sec)');
ylabel('Magnetic dipole (Am^2)');
title('Magnetic Dipole over Time');

%save('matfiles/trajOptSettings.mat', 'J'); 
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