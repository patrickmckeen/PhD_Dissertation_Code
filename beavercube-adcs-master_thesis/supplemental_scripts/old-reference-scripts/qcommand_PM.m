%r_ECI_axistoalign is the body axis of BeaverCube that you want to align
%with r_ECI (this axis will be aligned to 180 deg from earth-pointing axis)

%v_ECI_axistoalign is the body axis of BeaverCube that you want to align with v_ECI
%(the orbital direction)

%r_ECI_axistoalign, v_ECI_axistoalign are given in body coordinates
function q_command = qcommand_PM(v1, v2, u1, u2)
    arguments
        v1 (3,1)
        v2 (3,1)
        u1 (3,1)
        u2 (3,1)
    end
    %The approach is to find two axis-angle rotations: aligning r_ECI with
    %r_ECI_axistoalign, and then aligning v_ECI with v_ECI_axistoalign
    
    %this produces quaternion that rotates v onto u.
    u1 = u1/norm(u1);
    u2 = u2/norm(u2);
    v1 = v1/norm(v1);
    v2 = v2/norm(v2);
    
    %make sure the second (less important) axes are perpendicular to the
    %first
    v2 = v2 - dot(v1,v2)*v1;
    v2 = v2/norm(v2);
    
    u2 = u2 - dot(u1,u2)*u1;
    u2 = u2/norm(u2);
    
    q2 = vectors2q(v2,u2);
    
    %with this rotation, v2 is aligned with u2
    v1rot = quat_rotate(q2,v1);
    
    %now find rotation to align this with u1
    q1 = vectors2q(v1rot,u1);
    q_command = quat_multiply(q1, q2);
    
    
    
    
    %q_align_rECI describes the r_ECI_axistoalign -> r_ECI rotation (body
    %-> ECI)
%     q_align_rECI = vectors2q(r_ECI, transpose(r_ECI_axistoalign));
    
    %This step is necessary to ensure v_ECI as used to compute the v_ECI
    %alignment quaternion is perpendicular to r_ECI and that the rotations
    %can thus be performed in order without the second messing up the first
    
    %q_int_vr describes the r_ECI -> v_ECI rotation (body -> body)
%     q_int_vr = vectors2qangle(v_ECI, r_ECI, pi/2);
    %rotate r_ECI by q_int_vr to tranform r_ECI to v_ECI
%     rotated_vECI = quat_rotate(q_int_vr, r_ECI);
    
    %Now, we can find the v_ECI_axistoalign -> v_ECI rotation (body -> ECI)
    
    %rotate v_ECI_axis_toalign by q_align_r_ECI so we know where it is
    %after the r_ECI rotation
%     rotated_vECI_toalign = quat_rotate(q_align_rECI, transpose(v_ECI_axistoalign));
    
    %q_align_vECI describes the v_ECI_axistoalign -> v_ECI rotation (body
    %-> ECI), using v_ECI_axis_toalign after it has been rotated by the
    %first axis-angle rotation
%     q_align_vECI = vectors2q(rotated_vECI, rotated_vECI_toalign);
    
    %Multiply the two quaternions to find q_command
%     q_command = quat_multiply(q_align_rECI, q_align_vECI);
    
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
    rotated_vector = A*vec;
end
function q = vectors2q(vec_1, vec_2)
    %vec1 onto vec2
    vec_1 = vec_1(:)/norm(vec_1);
    vec_2 = vec_2(:)/norm(vec_2);
%     q_first = zeros(4,1);
%     axis_first = zeros(1,3);
    axis_first = cross(vec_1, vec_2);
    cos_angle = dot(vec_1, vec_2);
    if norm(axis_first)>0
        axis_first = axis_first/norm(axis_first);
    %     angle_first = 0;
    %     sin_angle = norm(axis_first);
        angle_first = acos(cos_angle);
    %     q_first(1) = cos(angle_first/2);
        q = [cos(angle_first/2);sin(angle_first/2)*axis_first];
    %     q_first(2) = sin(angle_first/2)*axis_first(1);
    %     q_first(3) = sin(angle_first/2)*axis_first(2);
    %     q_first(4) = sin(angle_first/2)*axis_first(3);
    %     q = q_first
    elseif cos_angle > 0
        q = [1 0 0 0].';
    elseif cos_angle < 0
        q = [0 1 0 0].';
    else
        error('how did you get here?')
    end
end
function q = vectors2qangle(vec_1, vec_2, angle)
    vec_1 = vec_1(:)/norm(vec_1);
    vec_2 = vec_2(:)/norm(vec_2);
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
    q = q_first
end