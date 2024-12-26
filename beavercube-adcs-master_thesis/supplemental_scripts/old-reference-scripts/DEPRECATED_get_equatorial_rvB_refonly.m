%%EQUATORIAL ORBIT TEST
sim_length = 86399; %1 day
% [r_init, v_init] = kep_ECI(0, 6771*1000, 0, 0, 0, 90, 0); %m and m/s
% r_init = r_init/1000; %km
% v_init = v_init/1000; %km/s
% [r_ECI_equatorial, B_ECI_equatorial, v_ECI_equatorial] = predict_Br(r_init, v_init, 1, sim_length, 1, 1); %km, km/s
% r_ECI_equatorial = r_ECI_equatorial*10^3; %m
% v_ECI_equatorial = v_ECI_equatorial*10^3; %m/s
% save('matfiles/r_ECI_equatorial.mat', 'r_ECI_equatorial');
% save('matfiles/v_ECI_equatorial.mat', 'v_ECI_equatorial');
% save('matfiles/B_ECI_equatorial.mat', 'B_ECI_equatorial');
% qf_desired = zeros(sim_length, 4);
%DEFINE COMMAND MODULE CONSTANT PARAMETERS%
% r_ECI_axistoalign = [1 0 0];
% v_ECI_axistoalign = [0 1 0];
% for i=1:sim_length
%     q_command = command(r_ECI_equatorial(i, :).', v_ECI_equatorial(i, :).', r_ECI_axistoalign, v_ECI_axistoalign);
%     if i > 1
%         q_prev = qf_desired(i-1,:);
%         sign_prev = sign(q_prev);
%         sign_current = sign(q_command);
%         if abs(q_prev(1) - q_command(1)) > 0.1 || abs(q_prev(2)-q_command(2)) > 0.2 || abs(q_prev(3)-q_command(3)) > 0.2 || abs(q_prev(4)-q_command(4)) > 0.2
%             q_command = q_command*-1;
%         end
%     end
%     qf_desired(i, :) = q_command;
% end
% qf_desired_equatorial = qf_desired;
% plot(qf_desired_equatorial(1:5520, :))
% save('matfiles/qf_desired_equatorial.mat', 'qf_desired_equatorial');
load_system('controller_testbed');
simOut = sim('controller_testbed');
xk_pred = simOut.yout{3}.Values.Data;
xk_est = xk_pred(200,1:7);
xk_est_minus = xk_pred(199,1:7);
xk_est_minus2 = xk_pred(198, 1:7);
q_pred = transpose(xk_est(1:4));
w_pred = transpose(xk_est(5:7));
q_pred_minus = xk_est_minus(1:4).';
w_pred_minus = xk_est_minus(5:7).';
w_predp = xk_est_minus2(5:7).';
qpred_store = zeros(299, 4);
qpred_disc_store = zeros(299, 4);
q_pred_disc = q_pred;
w_pred_disc = w_pred;
q1 = q_pred_disc(1);
q2 = q_pred_disc(2);
q3 = q_pred_disc(3);
q4 = q_pred_disc(4);
R_pred = [q1^2+q2^2-q3^2-q4^2 2*(q2*q3-q1*q4) 2*(q2*q4+q1*q3);...
          2*(q2*q3+q1*q4) q1^2-q2^2+q3^2-q4^2 2*(q3*q4-q1*q2);...
          2*(q2*q4-q1*q3) 2*(q3*q4+q1*q2) q1^2-q2^2-q3^2+q4^2];
q1 = q_pred_minus(1);
q2 = q_pred_minus(2);
q3 = q_pred_minus(3);
q4 = q_pred_minus(4);
R_predm = [q1^2+q2^2-q3^2-q4^2 2*(q2*q3-q1*q4) 2*(q2*q4+q1*q3);...
           2*(q2*q3+q1*q4) q1^2-q2^2+q3^2-q4^2 2*(q3*q4-q1*q2);...
           2*(q2*q4-q1*q3) 2*(q3*q4+q1*q2) q1^2-q2^2-q3^2+q4^2];
w_predm = w_pred_minus;
w_pred = (w_predm+w_pred+w_predp)/3;
for j = 1:30
    [R_pred, w_pred] = exponential_dyn(R_pred, w_pred, transpose([0 0 0]), J, 10);
     %bx = A*transpose([1 0 0]);
     q_pred = -1*transpose(dcm2quat(transpose(R_pred)));
     qpred_store(j,:) = q_pred;
     [R_predm, w_predm] = exponential_dyn(R_predm, w_predm, transpose([0 0 0]), J, 10);
     q_predm = -1*transpose(dcm2quat(transpose(R_predm)));
end
w_predf = (w_pred+w_predm)/2;
q_predf = (q_pred + q_predm);
q_predf = q_predf/norm(q_predf);
x_init_altro = [q_predf; w_predf];
v_ECI_equatorial_altro = v_ECI(500:end, :);
r_ECI_equatorial_altro = r_ECI(500:end, :);
B_ECI_equatorial_altro = B_ECI(500:end, :);
qf_desired_altro = qf_desired(500:end, :);
save('matfiles/julia_altro_xinit.mat', 'x_init_altro');
save('matfiles/julia_altro_v.mat', 'v_ECI_equatorial_altro');
save('matfiles/julia_altro_r.mat', 'r_ECI_equatorial_altro');
save('matfiles/julia_altro_b.mat', 'B_ECI_equatorial_altro');
save('matfiles/julia_altro_qf.mat', 'qf_desired_altro');
K_test = zeros(3, 6, 4100);
X_test = zeros(8, 4600);
U_test = zeros(3, 4600);
for i=1:4100
    if i < 501
        K_test(:, :, i) = zeros(3, 6);
    elseif i < 4100
        K_test(:, :, i) = K(:, :, i-500);
    else
        K_test(:, :, i) = zeros(3, 6);
    end
end
X_test = [zeros(8, 500) X];
U_test = [zeros(3, 500) U zeros(3, 1)];
function [r, rdot] = kep_ECI(eccentricity, a, inc, raan, arg_peri, true_anomaly ,t0)
%this function converts keplerian orbital elements and a mean anomaly to
%cartesian coordinates

%NOTE: Must have a in meters, measured from center of earth
    global G M
    %assumes all keplerian coordinates are in degrees
    true_anomaly_temp = rem(true_anomaly + t0*sqrt(G*M./a^3), 360);

    %Use Newton Rhapson to find Eccentric Anomaly
    E=zeros(1,101);
    E(1)=true_anomaly_temp/180*pi;
    for i=1:100
        E(i+1)=E(i)-(E(i)-eccentricity.*sin(E(i))-true_anomaly_temp/180*pi)./(1-eccentricity.*cos(E(i)));
    end

    %find true anomaly
    nu=2*atan2d(sqrt(1+eccentricity).*sin(E(end)/2),sqrt(1-eccentricity).*cos(E(end)/2));

    %find distance to central body
    r_c=a.*(1-eccentricity.*cos(E(end)));

    %find orbital position and velocity
    o=r_c*[cosd(nu) sind(nu) 0];
    o_dot=sqrt(G*M*a)/r_c*[-sin(E(end)) sqrt(1-eccentricity^2)*cos(E(end)) 0];

    %transform into cartesian
    r = R_z(-raan)*R_x(-inc)*R_z(-arg_peri)*o';
    rdot = R_z(-raan)*R_x(-inc)*R_z(-arg_peri)*o_dot';
end

function Rz = R_z(angle)

    Rz = [cosd(angle) sind(angle) 0;
            -sind(angle) cosd(angle) 0;
            0 0 1];
end

function Rx = R_x(angle)

    Rx = [1 0 0;
            0 cosd(angle) sind(angle);
            0 -sind(angle) cosd(angle)];
end
%Functions for orbit prop and IGRF
function [r_predict, B_predict, v_predict] = predict_Br(r_init, v_init, dt, slew_length, t_start, verbose)
    B_predict = zeros(slew_length/dt, 3);
    v_predict = zeros(slew_length/dt, 3);
    r_predict = zeros(slew_length/dt, 3);
    r_prev = r_init;
    v_prev = v_init;
    for i = 1:dt:slew_length
        if i > 1
            r_prev = r_predict(i-1,:);
            v_prev = v_predict(i-1,:); 
        end
        if i > 1 && mod(i, 50) == 0 && verbose == 1
            fprintf('%f %s of the way done calculating B, r, and v! \n', i*100/slew_length, '%');  
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
function q_command = command(r_ECI, v_ECI, r_ECI_axistoalign, v_ECI_axistoalign)
    %The approach is to find two axis-angle rotations: aligning r_ECI with
    %r_ECI_axistoalign, and then aligning v_ECI with v_ECI_axistoalign
    
    %q_align_rECI describes the r_ECI_axistoalign -> r_ECI rotation (body
    %-> ECI)
    r_ECI = r_ECI/norm(r_ECI);
    v_ECI = v_ECI/norm(v_ECI);
    q_align_rECI = vectors2q(r_ECI, transpose(r_ECI_axistoalign));
    
    %This step is necessary to ensure v_ECI as used to compute the v_ECI
    %alignment quaternion is perpendicular to r_ECI and that the rotations
    %can thus be performed in order without the second messing up the first
    
    %q_int_vr describes the r_ECI -> v_ECI rotation (body -> body)
    q_int_vr = vectors2qangle(v_ECI, r_ECI, pi/2);
    %rotate r_ECI by q_int_vr to tranform r_ECI to v_ECI
    rotated_vECI = quat_rotate(q_int_vr, r_ECI);
    
    %Now, we can find the v_ECI_axistoalign -> v_ECI rotation (body -> ECI)
    
    %rotate v_ECI_axis_toalign by q_align_r_ECI so we know where it is
    %after the r_ECI rotation
    rotated_vECI_toalign = quat_rotate(q_align_rECI, transpose(v_ECI_axistoalign));
    
    %q_align_vECI describes the v_ECI_axistoalign -> v_ECI rotation (body
    %-> ECI), using v_ECI_axis_toalign after it has been rotated by the
    %first axis-angle rotation
    q_align_vECI = vectors2q(rotated_vECI, rotated_vECI_toalign);
    
    %Multiply the two quaternions to find q_command
    q_command = quat_multiply(q_align_rECI, q_align_vECI);
    
    %Should be equal--checks
    
    %r_ECI/norm(r_ECI)
    %quat_rotate(q_command, transpose(r_ECI_axistoalign))
    
    %v_ECI/norm(v_ECI)
    %quat_rotate(q_command, transpose(v_ECI_axistoalign))
end
function q_out = quat_multiply(p, q)
    p1 = p(1);
    q1 = q(1);
    p_vec = p(2:4);
    q_vec = q(2:4);
    q_out = zeros(4,1);
    q_out(1) = p1*q1 - dot(p_vec, q_vec);
    q_out(2:4) = p1*q_vec + q1*p_vec + cross(p_vec, q_vec);
end
function rotated_vector = quat_rotate(q, vec)
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    q4 = q(4);
    A = [q1^2+q2^2-q3^2-q4^2 2*(q2*q3-q1*q4) 2*(q2*q4+q1*q3);...
        2*(q2*q3+q1*q4) q1^2-q2^2+q3^2-q4^2 2*(q3*q4-q1*q2);...
        2*(q2*q4-q1*q3) 2*(q3*q4+q1*q2) q1^2-q2^2-q3^2+q4^2];
    rotated_vector = transpose(A)*vec;
end
function q = vectors2q(vec_1, vec_2)
    vec_1 = vec_1/norm(vec_1);
    vec_2 = vec_2/norm(vec_2);
    q_first = transpose([0 0 0 0]);
    axis_first = zeros(1,3);
    axis_first = cross(transpose(vec_1), transpose(vec_2));
    if norm(axis_first)>0
        axis_first = axis_first/norm(axis_first);
    end
    angle_first = 0;
    cos_angle = dot(vec_1, vec_2)/(norm(vec_1)*norm(vec_2));
    if cos_angle > 1
        cos_angle = 1;
    end
    %sin_angle = norm(axis_first);
    %tan_angle = sin_angle/cos_angle;
    %angle_first = atan2(sin_angle, cos_angle);
    angle_first = acos(cos_angle);
    q_first(1) = cos(angle_first/2);
    q_first(2) = sin(angle_first/2)*axis_first(1);
    q_first(3) = sin(angle_first/2)*axis_first(2);
    q_first(4) = sin(angle_first/2)*axis_first(3);
    q = q_first;
end
function q = vectors2qangle(vec_1, vec_2, angle)
    vec_1 = vec_1/norm(vec_1);
    vec_2 = vec_2/norm(vec_2);
    q_first = transpose([0 0 0 0]);
    axis_first = zeros(1,3);
    axis_first = cross(transpose(vec_1), transpose(vec_2));
    if norm(axis_first)>0
        axis_first = axis_first/norm(axis_first);
    end
    angle_first = 0;
    %cos_angle = dot(vec_1, vec_2);
    %sin_angle = norm(axis_first);
    angle_first = angle;
    q_first(1) = cos(angle_first/2);
    q_first(2) = sin(angle_first/2)*axis_first(1);
    q_first(3) = sin(angle_first/2)*axis_first(2);
    q_first(4) = sin(angle_first/2)*axis_first(3);
    q = q_first;
end
function [R, w] = exponential_dyn(R, w, u, J, dt)
%     q1 = q_pred(1);
%     q2 = q_pred(2);
%     q3 = q_pred(3);
%     q4 = q_pred(4);
%     A_body2ECI = [q1^2+q2^2-q3^2-q4^2 2*(q2*q3-q1*q4) 2*(q2*q4+q1*q3);...
%                   2*(q2*q3+q1*q4) q1^2-q2^2+q3^2-q4^2 2*(q3*q4-q1*q2);...
%                   2*(q2*q4-q1*q3) 2*(q3*q4+q1*q2) q1^2-q2^2-q3^2+q4^2]
    %A_body2ECI = quat2dcm(transpose(q_pred));
    A_body2ECI = R;
    bx = A_body2ECI*transpose([1 0 0]);
    by = A_body2ECI*transpose([0 1 0]);
    bz = A_body2ECI*transpose([0 0 1]);
    %translate w into ECI
    w_body = A_body2ECI*w;
    %Find matrix exponential
    what = w_body/norm(w_body);
    theta = norm(w_body)*dt;
    skew_what = [0 -what(3) what(2)
                what(3) 0 -what(1)
                -what(2) what(1) 0];
    matrix_exponential = eye(3) + sin(theta)*skew_what + (1-cos(theta))*((skew_what)^2);
    %Update body x and y axes expressed in ECI using matrix exponential
    bx_new = matrix_exponential*bx;
    by_new = matrix_exponential*by;
    bz_new = matrix_exponential*bz;
    %Find new quaternion given x and y body axes expressed in ECI frame
    R = [bx_new by_new bz_new];
    R*transpose([1 0 0]);
    %q = transpose(dcm2quat(R));
    %Update w
    wdot = inv(J)*(u-cross(w, J*w));
    w = w + wdot*dt;
end