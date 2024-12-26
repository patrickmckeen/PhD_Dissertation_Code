close all
sim_length = 4100;
controller_status_arr = [0; 1; 2];
q_error_from_nadir_global = zeros(sim_length, 3);
w_norm_global = zeros(sim_length, 3);
for global_ind=1:3
    controller_status = controller_status_arr(global_ind);
    load_system('controller_testbed');
    simOut = sim('controller_testbed');
    q_error_from_nadir = zeros(sim_length, 1);
    norm_w = zeros(sim_length,1);
    x_true_raw = simOut.yout{2}.Values.Data;
    x_true = zeros(sim_length, 10);
    for ind=1:sim_length
        x_true(ind, :) = x_true_raw(:, 1, ind).';
    end
    for ind=500:sim_length
        true_q = x_true(ind, 1:4).';
        desired_q = qf_desired(ind, :).';
        q_error_from_nadir(ind) = quat_error(true_q, desired_q);
        norm_w(ind) = norm(x_true(ind, 5:7));
    end
    q_error_from_nadir_global(:, global_ind) = q_error_from_nadir.';
    w_norm_global(:, global_ind) = norm_w.';
end
t = [1:1:3601];
figure()
hold on
plot(t/60,q_error_from_nadir_global(500:end, 1))
plot(t/60,q_error_from_nadir_global(500:end, 2))
plot(t/60,q_error_from_nadir_global(500:end, 3))
legend('No control','B-dot control','TVLQR control');
xlabel('Time elapsed (minutes)');
ylabel('Control error angle (degrees)');
title('Controller Comparison for Equatorial Orbit: Quaternion Error over 1 hour');
hold off
figure()
hold on
plot(t/60,rad2deg(w_norm_global(500:end, 1)))
plot(t/60,rad2deg(w_norm_global(500:end, 2)))
plot(t/60,rad2deg(w_norm_global(500:end, 3)))
legend('No control','B-dot control','TVLQR control');
xlabel('Time elapsed (minutes)');
ylabel('Angular velocity (deg/sec)');
title('Controller Comparison for Equatorial Orbit: Angular Velocity over 1 hour');
hold off
% figure()
% plot(rad2deg(norm_w(500:end)))
function [error] = quat_error(qa, qb)
    qc = [qa(2:4); qa(1)];
    qc_rotate = [qc(4) qc(3) -qc(2) -qc(1);
                 -qc(3) qc(4)  qc(1) -qc(2);
                  qc(2) -qc(1) qc(4) -qc(3);
                  qc(1)  qc(2)  qc(3)  qc(4)];
    q_error = qc_rotate*([qb(2:4); qb(1)]);
    error = rad2deg(2*acos(q_error(4)));
    if error > 180
        error = 360-error;
    end
end
