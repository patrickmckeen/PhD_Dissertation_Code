close all
t_start = 300;
%%x_true
x_true_raw = simOut.yout{2}.Values.Data;
x_true = zeros(10, 4101);
for ind=1:4101
    x_true(:, ind) = x_true_raw(:, 1, ind);
end

u_true_raw = simOut.yout{4}.Values.Data;
u_true = zeros(3, 4101);
for ind=1:4101
    u_true(:, ind) = u_true_raw(:, 1, ind);
end
%%x_traj
X = load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrXset.mat').Xset;
load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrUset.mat');
%class(Xset)
%X = Xset;

%%x_predict
xk_predict = simOut.yout{3}.Values.Data;

%m
dipole = simOut.yout{10}.Values.Data;

%u
u_raw = simOut.yout{4}.Values.Data;
u = zeros(3, 4101);
for ind=1:4101
    u(:, ind) = u_raw(:, 1, ind);
end

%x_traj
x_traj = simOut.yout{14}.Values.Data;
%close all
figure()
hold on
plot(x_true(4:7, 301:3900).')
plot(X(4:7, 1:3599).')
hold off
figure()
hold on
plot(x_true(4:7, 301:1301).')
plot(xk_predict(301:1301, 1:4))
hold off
figure()
plot(x_true(1:3, 301:1301).')
figure()
hold on
plot(dipole(301:1301,:))
plot(Uset(:,1:1000).')
hold off

%close all
% figure()
% plot(Uinit.U.');
% figure()
% plot(Xtraj.');
% figure()
% plot(Uset.');
% figure()
% plot(Xset.');
load("/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/v_ECI.mat");
Xset = [x_true(1:3, 301:3900); x_true(4:7, 301:3900)];
size_Xset = size(Xset);
length_traj = size_Xset(2);
norm_vECI = zeros(length_traj, 3);
for ind=1:length_traj
    v_ind = v_ECI(t_start+ind, :);
    norm_vECI(ind, :) = -v_ind/norm(v_ind);
end

satAlignVector1 = [0; 0; -1];
prop_point = quatrotate(quatconj(Xset(4:7,:).'),satAlignVector1.');
prop_point_goal = quatrotate(quatconj(X(4:7,:).'),satAlignVector1.');
figure()
hold on
plot(prop_point);
plot(norm_vECI);
legend('Ux', 'Uy', 'Uz', '-Vx', '-Vy', '-Vz');
xlabel('Time elapsed (sec)');
ylabel('Position in ECI (normalized)');
title('Alignment of Propulsion System (U) with Anti-Velocity direction (-V), in ECI coordinates, with perfect knowledge of quaternion only');
hold off
figure()
plot(rad2deg(Xset(1:3,:).'))
xlabel('Time elapsed (sec)');
ylabel('Angular velocity (deg/s)');
title('Angular Velocity over Time, with perfect knowledge of quaternion only');

angle_error = zeros(length_traj, 1);
for ind=1:length_traj
    angle_error(ind) = acosd(dot(prop_point(ind, :), norm_vECI(ind, :)));
end
angle_error_trajOpt = zeros(length_traj-1, 1);
for ind=1:length_traj-1
    angle_error_trajOpt(ind) = acosd(dot(prop_point(ind, :), prop_point_goal(ind, :)));
end

load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrKset_lqr.mat');
K_lqr = zeros(3, 6, length_traj-1);
for ind=1:length_traj-1
    K_lqr(:, :, ind) = reshape(Kset_lqr(:, ind), 3, 6);
end

x_err = zeros(length_traj-1, 6);
u_err = zeros(length_traj-1, 3);
for ind=1:length_traj-1
    q_desired = X(4:7, ind);
    w_desired = X(1:3, ind);
    q_est = x_true(4:7, ind+299);
    w_est = x_true(1:3, ind+299);
    e = zeros(4, 1);
    e(1) = q_desired(1)*q_est(1) + q_desired(2:4).'*q_est(2:4);
    e(2:4) = q_desired(1)*q_est(2:4) - q_est(1)*q_desired(2:4) - cross(q_desired(2:4), q_est(2:4));
    phi = e(2:4)/(1+e(1));
    x_err(ind, :) = ([(w_est-w_desired); (phi*2)]).';
    u_err(ind,:) = (K_lqr(:, :, ind)*[(w_est-w_desired); (phi*2)]).';
end
figure()
plot(x_err)
figure()
plot(u_err)

x_err_raw = simOut.yout{15}.Values.Data;
figure()
plot(x_err_raw)

figure()
maxK = zeros(1, 3599);
minK = zeros(1, 3599);
meanK = zeros(1, 3599);
for i=1:3599
    maxK(i) = max(max(K_lqr(:, :, i)));
    minK(i) = min(min(K_lqr(:, :, i)));
    meanK(i) = mean(mean(K_lqr(:, :, i)));
end
hold on
plot(maxK)
plot(minK)
plot(meanK)
hold off



    
figure()
plot(angle_error)
xlabel('Time elapsed (sec)');
ylabel('Angle error (deg)');
title_str ="Angle between Propulsion System and Anti-Velocity direction" + newline +" in degrees, with perfect knowledge of quaternion only"; 
title(title_str);

figure()
plot(angle_error_trajOpt)
xlabel('Time elapsed (sec)');
ylabel('Angle error (deg)');
title('Deviation from Optimal Trajectory, in degrees, with perfect knowledge of quaternion only');

figure()
plot(dipole(301:3900,:));
xlabel('Time elapsed (sec)');
ylabel('Magnetic dipole (Am^2)');
title('Magnetic Dipole over Time, with perfect knowledge of quaternion only');