load("/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/v_ECI.mat");
load("/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/r_ECI.mat");
sim_length = 5520;
r_ECI_axistoalign = [1 0 0];
v_ECI_axistoalign = [0 1 0];
for i=1:sim_length
    q_command = command(r_ECI(i, :).', v_ECI(i, :).', r_ECI_axistoalign, v_ECI_axistoalign);
    if i > 1
        q_prev = qf_desired(i-1,:);
        sign_prev = sign(q_prev);
        sign_current = sign(q_command);
        if abs(q_prev(1) - q_command(1)) > 0.1 || abs(q_prev(2)-q_command(2)) > 0.2 || abs(q_prev(3)-q_command(3)) > 0.2 || abs(q_prev(4)-q_command(4)) > 0.2
            q_command = q_command*-1;
        end
    end
    qf_desired(i, :) = q_command;
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