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
load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrXset.mat');
load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrUset.mat');
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

figure()
plot(Xset(4:7, :).')