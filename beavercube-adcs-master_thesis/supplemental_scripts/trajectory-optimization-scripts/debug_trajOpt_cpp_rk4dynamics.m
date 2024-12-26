close all
t_start = 300;
t_end = 3898;
X = load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrXset.mat').Xset;
U = load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrUset.mat').Uset;
B = load('../clean-rpi/beavercube-adcs/bc/matfiles/B_ECI.mat').B_ECI;
K = load('../clean-rpi/beavercube-adcs/bc/matfiles/repackAlilqrKset_lqr.mat').Kset_lqr;

X_sim = zeros(7, 3599);
U_sim = zeros(3, 3599);

global J 
%Dynamics
J = [31364908.06, -6713613.57, 58830.40;
    -6713613.57,10040919.97,-123347.56;
    58830.40,-123347.56,34091278.27]*10^-9; %kg*m^2

X_sim(:, 1) = X(:, 1);
%randw = [rand()-0.5; rand()-0.5; rand()-0.5]*0.001;
%randq = [rand()-0.5; rand()-0.5; rand()-0.5; rand()-0.5]*0.01;
%X_sim(:, 1) = X_sim(:, 1) + [randw; randq];%[randw; 0; 0; 0; 0];
%X_sim(4:7, 1) = X_sim(4:7, 1)/norm(X_sim(4:7, 1));
for ind=1:3597
    %xk = X(:, ind-t_start+1);
    xk = X_sim(:, ind);
    Kk = reshape(K(:, ind), 3, 6);
    
    x_desired = X(:, ind);
    q_desired = x_desired(4:7);
    w_desired = x_desired(1:3);
    q_est = xk(4:7);
    w_est = xk(1:3);
    e = zeros(4, 1);
    e(1) = q_desired(1)*q_est(1) + q_desired(2:4).'*q_est(2:4);
    e(2:4) = q_desired(1)*q_est(2:4) - q_est(1)*q_desired(2:4) - cross(q_desired(2:4), q_est(2:4));
    phi = e(2:4)/(1+e(1));
    x_err = [(w_est-w_desired); (phi*2)];
    
    uk = U(:, ind)-Kk*x_err;
    uk(1) = min(0.2, uk(1));
    uk(1) = max(-0.2, uk(1));
    uk(2) = min(0.2, uk(2));
    uk(2) = max(-0.2, uk(2));
    uk(3) = min(0.2, uk(3));
    uk(3) = max(-0.2, uk(3));
    
    Bk = B(ind+300, :).';
    xkp1 = rk4(xk, uk, Bk);%rk4(xk, uk, Bk);
    if ind==1
        cross(rotT(xk(4:7))*Bk,uk)
    end
    xkp1(4:7) = xkp1(4:7)/norm(xkp1(4:7));
    X_sim(:, ind+1) = xkp1;
    U_sim(:, ind) = uk;
end

figure()
hold on
plot(X.')
plot(X_sim.')
hold off



figure()
plot(U_sim.')


function Rt = rotT(q) %ECI->body, vbody = R*vECI
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    q4 = q(4);
    
    Rt = [q1^2+q2^2-q3^2-q4^2 2*(q2*q3-q1*q4) 2*(q2*q4+q1*q3);...
        2*(q2*q3+q1*q4) q1^2-q2^2+q3^2-q4^2 2*(q3*q4-q1*q2);...
        2*(q2*q4-q1*q3) 2*(q3*q4+q1*q2) q1^2-q2^2-q3^2+q4^2].';
end

function vx = skewsym(v)
    vx=[0 -v(3) v(2) ; v(3) 0 -v(1) ; -v(2) v(1) 0 ];
end

function W = Wmat(q)
    W = [-q(2:end).'; q(1)*eye(3,3) + skewsym(q(2:end))];
end

function xdot = dynamics(x,u, B) %continuous dynamics
    global J
    w = x(1:3);
    q = x(end-3:end);
    wdot = -J\(cross(w,J*w) + cross(rotT(q)*B,u));
%     wdot = -J\(cross(w,J*w) + u/1000 - dtau);
    qdot = 0.5*Wmat(q)*w;
    xdot = [wdot;qdot];
end

function xdot = dynamicsNew(x,torque) %continuous dynamics
    global J
    w = x(1:3);
    q = x(end-3:end);
    wdot = -J\(cross(w,J*w) + torque);
%     wdot = -J\(cross(w,J*w) + u/1000 - dtau);
    qdot = 0.5*Wmat(q)*w;
    xdot = [wdot;qdot];
end


function x = discdyn(x, u, B, times)
    for ind=1:times
        %q_des = x_desired(4:7);
        %q = x(4:7);
%         torque_req = cross(rotT(q_des)*B,u);
%         B_body = rotT(q)*B;
%         torque = -cross(B_body,torque_req)/(norm(B_body)^2);
%         %torque = torque-Kk*x_err;
%         torque(1) = min(0.2, torque(1));
%         torque(1) = max(-0.2, torque(1));
%         torque(2) = min(0.2, torque(2));
%         torque(2) = max(-0.2, torque(2));
%         torque(3) = min(0.2, torque(3));
%         torque(3) = max(-0.2, torque(3));
        xdot = dynamics(x, u, B);
        x = x+xdot/times;
        x(4:7) = x(4:7)/norm(x(4:7));
    end
end

function xkp1 = rk4(xk,uk,Bk)
    k1 = dynamics(xk,uk,Bk);
    k2 = dynamics(xk+0.5*k1,uk,Bk);
    k3 = dynamics(xk+0.5*k2,uk,Bk);
    k4 = dynamics(xk+k3,uk,Bk);
    xkp1 = xk + (1/6)*(k1+2*k2+2*k3+k4);
end