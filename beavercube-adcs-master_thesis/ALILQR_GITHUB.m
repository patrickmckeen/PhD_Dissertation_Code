%Matlab altro

%% set up
%%%%%%%%%%%%MODIFY THESE PER PROBLEM
%currently does not allow for infeasible start and is not minimum time and
%is not nesterov

%n is number of elements in state, m is number of elements in control, N is
%number of time steps. quaternions should always be last 4 elements of
%state
simulation_initialize


% potentially useful prepared vals
uxMax = 0.15; %Am^2
uyMax = 0.15; %Am^2
uzMax = 0.15; %Am^2
global umax
umax =[uxMax;uyMax;uzMax];
global J
J = [31364908.06, -6713613.57, 58830.40;
    -6713613.57,10040919.97,-123347.56;
    58830.40,-123347.56,34091278.27]*10^-9; %kg*m^2

global gravitygradient_on aero_on srp_on dipole_on prop_on
global n m N numc numIneq numDOF
global x0
%GLOBAL VARIABLES ARE USED FOR PARAMETERS/CONSTANTS/ETC THAT DO NOT CHANGE
%DURING RUNTIME.
%there is basically no checking for consistent dimensions.
global dt
dt = 1; %s
global QN R 
global B_ECI
global swslew swpoint sv1 sv2 su sratioslew
Rset = r_ECI.'; 
x0 = [x_init(5:7);x_init(1:4)];
% x0 = [0;0;0;0.5;0.5;0.5;0.5];
x0(end-3:end) = x0(end-3:end)./norm(x0(end-3:end));
N = 1000;
global Nslew
Nslew = round(N/2);
global LA0
LA0 = 1/eps;

global rNslew
rNslew = Rset(:,dt*Nslew);

down = -Rset./vecnorm(Rset);
ram = v_ECI.'./vecnorm(v_ECI.');
qdes = nan(4,N+1);
wdes = nan(3,N+1);
%xaxis pointing nadir, z axis pointing ram

rotmatdes1 = [down(:,1) cross(ram(:,1),down(:,1)) ram(:,1)];
rotmatdes1 = rotmatdes1./vecnorm(rotmatdes1);
qdes(:,1) = rotm2quat(rotmatdes1);
for k = 2:N-1
    rotmatdes = [down(:,k) cross(ram(:,k),down(:,k)) ram(:,k)];
    rotmatdes = rotmatdes./vecnorm(rotmatdes);
    qk = rotm2quat(rotmatdes).';
    qdes(:,k) = qk;
    
    qkm1 = qdes(:,k-1);
    dq = quatdivide(qk.',qkm1.');
    ang = 2*acos(dq(1));
    vec = dq(2:end)/sqrt(1-dq(1)^2);
    
    wdes(:,k-1) = vec*ang/dt;
end
qdes = qdes(:,1:N);
wdes = wdes(:,1:N);
xdes = [wdes;qdes];


xNslew = xdes(:,Nslew);

aldes = (wdes(:,2:N)-wdes(:,1:N-1))/dt;
mdes = (J\aldes)./mean(vecnorm(B_ECI.'));

swslew = 0;%10/max(abs(wdes)+0.01,[],'all')^2;
swpoint = 1000;%100/max(abs(wdes)+0.01,[],'all')^2;
sv1 = 1e5;
sv2 = 1e5;
su = 1;%1/max(abs(mdes)+1e-6,[],'all');
sratioslew = 0.01;

% QN = Q(N,Rset(:,dt*N)); %assuming quaternions are part of state, this should be N-1 x N-1, symmetric, real, and positive semi-definite
QN = [swpoint*eye(3,3) zeros(3,3);zeros(3,3) sv2*eye(3,3)];
R = su*eye(3,3);


%exactly one of xf and trajDes should be empty
xf = [0;0;0;0.1;0.2;0.1;0.1]; %should be empty or n x 1
xf(end-3:end) = xf(end-3:end)./norm(xf(end-3:end));
xf = [];
trajDes =[repmat(xNslew,[1 Nslew]) xdes(:,Nslew+1:end)]; %should be empty or n x (N)

%physics constants
global grav_const m_earth r_earth
grav_const = 6.6742e-11; %m^3*kg^−1*s^−2
%Mass of Earth
m_earth = 5.9736e+24; %kg
%Radius of Earth
r_earth = 6361010; %m


load('matfiles/g_coefs.mat');
load('matfiles/h_coefs.mat');
global g
global h
    
%algorithm parameters
%shared params
global maxIter costTol
costTol = 1e-4;
maxIter = 500;

%ilqr params
global gradTol maxIlqrIter beta1 beta2 maxLsIter regInit regMax regMin regScale costMax regBump zcountLim ilqrCostTol maxControl maxState
gradTol = 1e-5;
ilqrCostTol = 50*costTol;
maxIlqrIter = 100; %250
beta1 = 1e-8;
beta2 = 10;
maxLsIter = 20;
regInit = 0;
regMax = 1e8;
regMin = 1e-8;
regBump = 10;
regScale = 1.6;
costMax = 1e8;
zcountLim = 5;
maxControl = 1e8; %this is NOT the constraint value. This is a number to throw an error if we see
maxState = 1e8;

%AL params
global cmax maxOuterIter penMax penScale penInit lagMultMax lagMultMin lagMultInit dJcountLim
cmax = 1e-3;
maxOuterIter = 30;
dJcountLim = 10;
penMax = 1e8;
penScale = 10;
penInit = 1;
lagMultMax = 1e8;
%lagMultMin = -1e4;%-1e8;
lagMultInit = 0;

%%
Xdesset = setup(x0,trajDes,xf);
%% run the code

[Xset,Uset,lambdaSet,Kset,Pset] = trajOpt(x0,Rset,Xdesset);
[Kset2,Sset] = TVLQRconst(Xset,Rset);

%% functions that define dynamics, metrics, constraints, etc. (These would change between BC and other satellites or different trajectory goals, disturbance torques, etc.)
function Qk = Q(k,rk)
    %assuming quaternions are part of state, this should be N-1 x N-1, symmetric, real, and positive semi-definite
    %this is nadir pointing for now
    global Nslew sv1 swpoint sv2 rNslew swslew sratioslew
    if k >= Nslew
        a = [0 0 0].';
        b = rk;
        e1 = -(b-a);
        e1 = e1/norm(e1);
        e2 = [1;0;0];
        if abs(dot(e2,e1)) > 0.9
            e2 = [0;1;0];
        end
        e2 = e2-dot(e2,e1);
        e2 = e2/norm(e2);
        e3 = cross(e1,e2);
        e3 = e3/norm(e3);
        Qkqq = eye(3,3)*sv1;%sv1*(e2*e2.') + sv2*(e3*e3.');
        Qkqw = zeros(3,3);
        Qkww = swpoint*eye(3,3);
        Qk = [Qkww Qkqw; Qkqw.' Qkqq];
    else
        Qk = Q(Nslew,rNslew)*sratioslew;
        Qk(1:3,1:3) = eye(3,3)*swslew;
    end
%      Qk = [0*eye(3,3) zeros(3,3);zeros(3,3) sv1*eye(3,3)];
end

function xdot = dynamics(x,u,r,t) %continuous dynamics
    global J
    w = x(1:3);
    q = x(end-3:end);
    B = magneticFieldECI(r,t);
    dtau = distTorq(r,x);
    wdot = -J\(cross(w,J*w) + cross(rotT(q)*B,u) - dtau);
    qdot = 0.5*Wmat(q)*w;
    xdot = [wdot;qdot];
end

function [jacX,jacU] = dynamicsJacobians(x,u,r,t) %these are the Jacobians of the continuous function. %assuming r (and thus B) is constant over timestep
    global J
    w = x(1:3);
    q = x(end-3:end);
    [Rt1, Rt2, Rt3, Rt4] = Rti(q);
    B = magneticFieldECI(r,t);
    
    [distTorqJacW,distTorqJacQ] = distTorqJac(r,x);
    jww = J\(skewsym(J*w) - skewsym(w)*J + distTorqJacW);
    jwq = J\(skewsym(u)*[Rt1*B Rt2*B Rt3*B Rt4*B]+ distTorqJacQ);
    jqw =  0.5*Wmat(q);
    jqq = 0.5*[0 -w.';w -skewsym(w)];
    jacX = [jww jwq; jqw jqq];
    
    jwu = -J\skewsym(rotT(q)*B);
    jqu = zeros(4,3);
    jacU = [jwu;jqu];
end


%%disturbance models
function pt = propTorq()
    pt = [0;0;1e-6];
end

function gt = ggTorq(r,x)
    global m_earth grav_const J
    q = x(end-3:end);
    rb = rotT(q)*r;
    gt = -(3*m_earth*grav_const/(norm(rb)^5))*cross(rb,J*rb);
end

function mt = magTorq(r,x)
    q = x(end-3:end);
    mt = [0;0;0];
end

function drt = dragTorq(r,x)
    q = x(end-3:end);
    drt = [0;0;0];
end

function st = srpTorq(r,x)
    q = x(end-3:end);
    st = [0;0;0];
end

function [jptw,jptq] = jacPropTorq()
    jptw = zeros(3,3);
    jptq = zeros(3,4);
end

function [jgtw,jgtq] = jacggTorq(r,x)
    global m_earth grav_const J
    q = x(end-3:end);
    rb = rotT(q)*r;
    [Rt1, Rt2, Rt3, Rt4] = Rti(q);
    jgtw = zeros(3,3);
    jgtq = (3*m_earth*grav_const/(norm(rb)^5))*(skewsym(rb)*J*[Rt1*r Rt2*r Rt3*r Rt4*r]-skewsym(J*rb)*[Rt1*r Rt2*r Rt3*r Rt4*r]);
end

function [jmtw,jmtq] = jacMagTorq(r,x)
    q = x(end-3:end);
    jmtw = zeros(3,3);
    jmtq = zeros(3,4);
end

function [jdrtw,jdrtq] = jacDragTorq(r,x)
    q = x(end-3:end);
    jdrtw = zeros(3,3);
    jdrtq = zeros(3,4);
end

function [jstw,jstq] = jacsrpTorq(r,x)
    q = x(end-3:end);
    jstw = zeros(3,3);
    jstq = zeros(3,4);
end

%constraints. Always return same nmber, numc, of constraints? if tehre are
%different constraints at different time steps, just return 0 or something
%for the extra ones so they are always met.
%same for number of inequality constraints, numIneq

function ce = eqConstraints(k,x,u)  %ineq before eq
    ce = [];
end

function ci = ineqConstraints(k,x,u)  %ineq before eq
    global N umax
    if k == N
        ci = zeros(6,1);
    elseif k<N && k >= 0
        ci = [u-umax;-u-umax];
    else
        error('invalid time index')
    end
end

function [ecku,eckx] = eqConstraintsJacobians(k,x,u)  %ineq before eq
    ecku = [];
    eckx = [];
end

function [icku,ickx] = ineqConstraintsJacobians(k,x,u)  %ineq before eq
    global N numc n m
    if k == N
        icku = zeros(numc,m);
        ickx = zeros(numc,n);
    elseif k<N && k>=0
        icku = [eye(3,3);-eye(3,3)];
        ickx = zeros(numc,n);
    else
        error('invalid time index')
    end
end

%% function def (these are constant)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% these probably don't change per
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% problem.

function dtau = distTorq(r,x)
    global gravitygradient_on aero_on srp_on dipole_on prop_on
    dtau = zeros(3,1);
    if gravitygradient_on
        dtau = dtau + ggTorq(r,x);
    end
    
    if aero_on
        dtau = dtau + dragTorq(r,x);
    end
    
    if srp_on
        dtau = dtau + srpTorq(r,x);
    end
    
    if dipole_on
        dtau = dtau + magTorq(r,x);
    end
    
    if prop_on
        dtau = dtau + propTorq();
    end
end

function [ jacw,jacq ] = distTorqJac(r,x)
    global gravitygradient_on aero_on srp_on dipole_on prop_on
    jacw = zeros(3,3);
    jacq = zeros(3,4);
    if gravitygradient_on
        [dtw,dtq] = jacggTorq(r,x);
        jacw = jacw + dtw;
        jacq = jacq + dtq;
    end
    
    if aero_on
        [dtw,dtq] = jacDragTorq(r,x);
        jacw = jacw + dtw;
        jacq = jacq + dtq;
    end
    
    if srp_on
        [dtw,dtq] = jacsrpTorq(r,x);
        jacw = jacw + dtw;
        jacq = jacq + dtq;
    end
    
    if dipole_on
        [dtw,dtq] = jacMagTorq(r,x);
        jacw = jacw + dtw;
        jacq = jacq + dtq;
    end
    
    if prop_on
        [dtw,dtq] = jacPropTorq();
        jacw = jacw + dtw;
        jacq = jacq + dtq;
    end   
end

function Xdesset = setup(x0,trajDes,xf)
    global numc numIneq m n N R QN numDOF
    numc = size(constraints(N,x0,zeros(1,m)),1);
    numIneq = size(ineqConstraints(N,x0,zeros(1,m)),1);
    m = length(R);
    
    if length(unique([size(QN) size(Q(x0,zeros(1,m)))])) > 1
        error('QN and Q must be same size and square')
    end
    numDOF = length(QN);
    if ~isempty(trajDes) && ~isempty(xf)
        error('cannot specify both trajDes and xf')
    elseif ~isempty(xf)
        Xdesset = xf*ones(1,N);
        n = length(xf);
    elseif ~isempty(trajDes)
        Xdesset = trajDes;
        xf = trajDes(:,end);
        n = size(trajDes,1);
        N = length(trajDes);
    else
        error('need some sort of goal--specify xf or trajDes')
    end
end


function c = constraints(k,x,u)
    c = [ineqConstraints(k,x,u);eqConstraints(k,x,u)];
end

function [cku,ckx] = constraintJacobians(k,x,u)
    [ecku,eckx] = eqConstraintsJacobians(k,x,u);
    [icku,ickx] = ineqConstraintsJacobians(k,x,u);
    cku = [icku;ecku];
    ckx = [ickx;eckx]*Gmat(x(end-3:end)).';
end


function [lkxx,lkuu,lkux,lkx,lku,ckx,cku] = costJacobians(k,xk,uk,xdesk,rk)
    global N R QN
    [cku,ckx] = constraintJacobians(k,xk,uk);
    if k == N
        lkuu = R*0;
        lku = uk*0;
        lkux = zeros(length(uk),length(xk))*Gmat(xk(end-3:end)).';
        
        lkx = QN*Gmat(xk(end-3:end))*(xk-xdesk);
        lkxx = QN;
    elseif k<N && k>=0
        lkuu = R;
        lku = R*uk;
        lkux = zeros(length(uk),length(xk))*Gmat(xk(end-3:end)).';
        
        lkx = Q(k,rk)*Gmat(xk(end-3:end))*(xk-xdesk);
        lkxx = Q(k,rk);
    else
        error('invalid time index')
    end
end

function B = magneticFieldECI(rECI,t)
    global B_ECI dt
    B = B_ECI(t,:).';
%     global g h
%     utc_time = get_utc(t);
%     r_LLA = eci2lla(rECI.', utc_time);
%     lat = r_LLA(1); %deg
%     long = r_LLA(2); %deg
%     alt = r_LLA(3); %m above sea level
%     colatitude = (90-lat);
%     east_long = long;
%     if east_long < 0
%         east_long = east_long + 360;
%     end
%     east_long = (east_long);
%     a = 6371.2*10^3;
%     alt = alt+a;
%     Br = 0;
%     Btheta = 0;
%     Bphi = 0;
%     dP = zeros(14,14);
%     for n=1:13
%         for m=0:n
%             P_n_minus_1 = legendre(n-1, cosd(colatitude), 'sch');
%             if n==m && n==1
%                 dP(n+1, m+1) = cosd(colatitude);
%             elseif n==m
%                 P_n_minus_n_minus = P_n_minus_1(n);
%                 dP_nn = sind(colatitude)*dP(n, n) + cosd(colatitude)*P_n_minus_n_minus;
%                 dP(n+1, m+1) = sqrt(1 - 1/(2*n))*dP_nn;
%             else
%                 dP_nn = (2*n - 1)/sqrt(n^2 - m^2)*(cosd(colatitude)*dP(n, m+1) - sind(colatitude)*P_n_minus_1(m+1));
%                 K_nm = 0;
%                 if n > 1
%                     K_nm = sqrt(((n-1)^2 - m^2)/(n^2 - m^2));
%                     dP_nn = dP_nn - K_nm*dP(n-1, m+1);
%                 end
%                 dP(n+1, m+1) = dP_nn;
%             end
%         end
%     end
%     for n=1:13
%         P = legendre(n, cosd(colatitude), 'sch');
%         Br_n = 0;
%         Btheta_n = 0;
%         Bphi_n = 0;
%         for m=0:n
%             g_nm = g(n, m+1);
%             h_nm = h(n, m+1);
%             P_nm = P(m+1);
%             dP_nm = dP(n+1, m+1);
%             Br_nm = (g_nm*cosd(m*east_long)+h_nm*sind(m*east_long))*P_nm;
%             Br_n = Br_n + Br_nm;
%             Btheta_nm = (g_nm*cosd(m*east_long)+h_nm*sind(m*east_long))*dP_nm;
%             Btheta_n = Btheta_n + Btheta_nm;
%             Bphi_nm = m*(-g_nm*sind(m*east_long)+h_nm*cosd(m*east_long))*P_nm;
%             Bphi_n = Bphi_n + Bphi_nm;
%         end
%         Br_n = Br_n*((a/alt)^(n+2))*(n+1);
%         Br = Br + Br_n;
%         Btheta_n = Btheta_n*((a/alt)^(n+2));
%         Btheta = Btheta + Btheta_n;
%         Bphi_n = Bphi_n*((a/alt)^(n+2));
%         Bphi = Bphi + Bphi_n;
%     end
%     Bphi = Bphi*-1/(sind(colatitude));
%     Br;
%     Btheta = Btheta*-1;
%     Bn = -Btheta;
%     Be = Bphi;
%     Bd = -Br;
%     B_ned = [Bn, Be, Bd];
%     [b_ecefx, b_ecefy, b_ecefz] = ned2ecefv(Bn, Be, Bd, lat, long);
%     B = lla2eci(ecef2lla([b_ecefx b_ecefy b_ecefz]),utc_time).';
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

function LA = cost(Xset,Uset,Rset,Xdesset,lambdaSet,mu)
    global N QN R dt
    LA = 0;
    k = N;
    xk = Xset(:,k);
    lamk = lambdaSet(:,k);
    dLA = 0;
    xdesk = Xdesset(:,k);
    uk = zeros(3,1);
    ck = constraints(k,xk,uk);
    Imuk = Imu(mu,ck,lamk);
    dLA = dLA + 0.5*(xk-xdesk).'*Gmat(xk(end-3:end)).'*QN*Gmat(xk(end-3:end))*(xk-xdesk);
    dLA = dLA + (lamk + 0.5*Imuk*ck).'*ck;
    LA = LA + dLA;
    for k = 1:N-1
        uk = Uset(:,k);
        xk = Xset(:,k);
        rk = Rset(:,k*dt);
        lamk = lambdaSet(:,k);
        xdesk = Xdesset(:,k);
        ck = constraints(k,xk,uk);
        Imuk = Imu(mu,ck,lamk);
        dLA = 0;
        dLA = dLA + 0.5*(xk-xdesk).'*Gmat(xk(end-3:end)).'*Q(k,rk)*Gmat(xk(end-3:end))*(xk-xdesk);
        dLA = dLA + 0.5*uk.'*R*uk;
        dLA = dLA + (lamk + 0.5*Imuk*ck).'*ck;
        LA = LA + dLA;
    end
end

function R = rot(q) %body->ECI, vECI = R*vbody
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    q4 = q(4);
    
    R = [q1^2+q2^2-q3^2-q4^2 2*(q2*q3-q1*q4) 2*(q2*q4+q1*q3);...
        2*(q2*q3+q1*q4) q1^2-q2^2+q3^2-q4^2 2*(q3*q4-q1*q2);...
        2*(q2*q4-q1*q3) 2*(q3*q4+q1*q2) q1^2-q2^2-q3^2+q4^2];
end

function [R1, R2, R3, R4] = Ri(q)
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    q4 = q(4);
   
    R1 = 2*[q1 q4 -q3; -q4 q1 q2; q3 -q2 q1];
    R2 = 2*[q2 q3 q4; q3 -q2 q1; q4 -q1 -q2];
    R3 = 2*[-q3 q2 -q1; q2 q3 q4; q1 q4 -q3];
    R4 = 2*[-q4 q1 q2; -q1 -q4 q3; q2 q3 q4];
end


function Rt = rotT(q) %ECI->body, vbody = R*vECI
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    q4 = q(4);
    
    Rt = [q1^2+q2^2-q3^2-q4^2 2*(q2*q3-q1*q4) 2*(q2*q4+q1*q3);...
        2*(q2*q3+q1*q4) q1^2-q2^2+q3^2-q4^2 2*(q3*q4-q1*q2);...
        2*(q2*q4-q1*q3) 2*(q3*q4+q1*q2) q1^2-q2^2-q3^2+q4^2].';
end

function [Rt1, Rt2, Rt3, Rt4] = Rti(q)
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    q4 = q(4);
    
    Rt1 = 2*[q1 q4 -q3; -q4 q1 q2; q3 -q2 q1].';
    Rt2 = 2*[q2 q3 q4; q3 -q2 q1; q4 -q1 -q2].';
    Rt3 = 2*[-q3 q2 -q1; q2 q3 q4; q1 q4 -q3].';
    Rt4 = 2*[-q4 q1 q2; -q1 -q4 q3; q2 q3 q4].';
end

function vx = skewsym(v)
    vx=[0 -v(3) v(2) ; v(3) 0 -v(1) ; -v(2) v(1) 0 ];
end

function W = Wmat(q)
    W = [-q(2:end).'; q(1)*eye(3,3) + skewsym(q(2:end))];
end

%%%%%%%%%%%%%%%%%
function xkp1 = rk4(xk,uk,rk,tk)
    global dt
    k1 = dynamics(xk,uk,rk,tk);
    k2 = dynamics(xk+0.5*dt*k1,uk,rk,tk);
    k3 = dynamics(xk+0.5*dt*k2,uk,rk,tk);
    k4 = dynamics(xk+dt*k3,uk,rk,tk);
    xkp1 = xk + (dt/6)*(k1+2*k2+2*k3+k4);
end

function Imu = Imu(mu,c,lam) %ineq before eq
    global numc numIneq cmax
    ii = mu*ones(1,numc);
    iszer = and(and((c <= 0),lam <= 0),(1:numc).'<= numIneq);
%     iszer = and(and((c <= -cmax),lam <= 0),(1:numc).'<= numIneq);
    ii(iszer) = 0;
    Imu = diag(ii);
end

function [Ak,Bk] = rk4Jacobians(xk,uk,rk,tk)
    global dt
    k1 = dynamics(xk,uk,rk,tk);
    k2 = dynamics(xk+0.5*dt*k1,uk,rk,tk);
    k3 = dynamics(xk+0.5*dt*k2,uk,rk,tk);
    k4 = dynamics(xk+dt*k3,uk,rk,tk);
    
    [F1,E1] = dynamicsJacobians(xk,uk,rk,tk);
    m1 = F1;
    n1 = E1;
    [F2,E2] = dynamicsJacobians(xk+0.5*dt*k1,uk,rk,tk);
    m2 = F2 + 0.5*dt*F2*m1;
    n2 = E2 + 0.5*dt*F2*n1;
    [F3,E3] = dynamicsJacobians(xk+0.5*dt*k2,uk,rk,tk);
    m3 = F3 + 0.5*dt*F3*m2;
    n3 = E3 + 0.5*dt*F3*n2;
    [F4,E4] = dynamicsJacobians(xk+dt*k3,uk,rk,tk);
    m4 = F4 + dt*F4*m3;
    n4 = E4 + dt*F4*n3;
    Ak = eye(length(xk)) + (dt/6)*(m1+2*m2+2*m3+m4);
    Bk = (dt/6)*(n1 + 2*n2 + 2*n3 + n4);
end

function G = Gmat(x)
    q = x(end-3:end);
    G = [eye(3,3) zeros(3,4);zeros(3,3) Wmat(q).'];
end

function [Pset,Kset,dset,delV,rho,drho] = backwardpass(Xset,Uset,Rset,Xdesset,lambdaSet,rho,drho,mu)
    global N m n QN regScale regMin dt numDOF
    Kset = nan(m,numDOF,N);
    Pset = nan(numDOF,numDOF,N);
    dset = nan(m,N-1);
    delV = [0 0];
    
    k=N;
    [lkxx,lkuu,lkux,lkx,lku,ckx,cku] = costJacobians(k,Xset(:,k),zeros(m,1),Xdesset(:,k),Rset(:,k*dt));
    ck = constraints(k,Xset(:,k),zeros(m,1));
    Imuk = Imu(mu,ck,lambdaSet(:,k));
    
    pk = lkx + ckx.'*(lambdaSet(:,k) + Imuk*ck);
    Pk = lkxx + ckx.'*Imuk*ckx;
  
    qk = Xset(end-3:end,end);
    Gk = Gmat(qk);
    
    pn = pk;
    Pn = Pk;
    Gn = Gk;
    Pset(:,:,k) = Pk;
    
    k = N-1;
    while k>0
        Gkp1 = Gk;
        Pkp1 = Pk;
        pkp1 = pk;
        
        xk = Xset(:,k);
        xdesk = Xdesset(:,k);
        uk = Uset(:,k);
        qk = xk(end-3:end);
        tk = k*dt;
        rk = Rset(:,tk);
        
        Gk = Gmat(qk);
        [Ak,Bk] = rk4Jacobians(xk,uk,rk,tk);
        Aqk = Gkp1*Ak*Gk.';
        Bqk = Gkp1*Bk;
        
        [lkxx,lkuu,lkux,lkx,lku,ckx,cku] = costJacobians(k,xk,uk,xdesk,rk);
        ck = constraints(k,xk,uk);
        Imuk = Imu(mu,ck,lambdaSet(:,k));
        Qkuu = lkuu + Bqk.'*Pkp1*Bqk + cku.'*Imuk*cku;
        Qkuureg = Qkuu + rho*eye(m,m);
        if cond(Qkuureg) > 50 %~all(real(eig(triu(Qkuureg) + triu(Qkuureg)'))) > 0
            %problem! Qkuu is not invertible??
            k = N-1;
            drho = max(drho*regScale,regScale);
            rho = max(rho*drho,regMin);
            
            pk = pn;
            Pk = Pn;
            Gk = Gn;
        else
            Qkxx = lkxx + Aqk.'*Pkp1*Aqk + ckx.'*Imuk*ckx;
            Qkux = lkux + Bqk.'*Pkp1*Aqk + cku.'*Imuk*ckx;
            Qkx = lkx + Aqk.'*pkp1 + ckx.'*(lambdaSet(:,k) + Imuk*ck);
            Qku = lku + Bqk.'*pkp1 + cku.'*(lambdaSet(:,k) + Imuk*ck);
            
            Kk = -Qkuureg\Qkux;
            Kset(:,:,k) = Kk;
            dk = -Qkuureg\Qku;
            dset(:,k) = dk;
            Pk = Qkxx + Kk.'*Qkuu*Kk + Kk.'*Qkux + Qkux.'*Kk;
            pk = Qkx + Kk.'*Qkuu*dk + Kk.'*Qku + Qkux.'*dk;
            Pk = 0.5*(Pk+Pk.');
            Pset(:,:,k) = Pk;
            delV = delV + [dk.'*Qku 0.5*dk.'*Qkuu*dk];
            k = k-1;
        end
        
    end
    drho = min(drho/regScale,1/regScale);
    rho = rho*drho*(rho*drho>regMin);
end

function [newXset,newUset,newLA,rho,drho] = forwardpass(Xset,Uset,Kset,Rset,dset,delV,LA,lambdaSet,rho,drho,mu,Xdesset)
    global N maxLsIter beta1 beta2 regScale regMin regBump maxState maxControl
    alph = 1;
    newLA = 1/eps;
    z = -1;
    
    iter = 0;
    while ((z<=beta1 || z>beta2) && (newLA>=LA))
        if iter > maxLsIter
            newXset = Xset;
            newUset = Uset;
            newLA = cost(newXset,newUset,Rset,Xdesset,lambdaSet,mu);
            z = 0;
            alph = 0;
            exp = 0;
            drho = max(drho*regScale,regScale);
            rho = max(rho*drho,regMin);
            rho = rho + regBump;
            break
        end
        
        [newXset,newUset] = generateTrajectory(Xset,Uset,Kset,dset,Rset,alph,lambdaSet);
        
        nonfinflag = any([any(isnan(newXset),'all'),any(isnan(newUset),'all'),any(isinf(newXset),'all'),any(isinf(newUset),'all'),any(newXset > maxState,'all'),any(newUset > maxControl,'all')]);
        if nonfinflag
            iter = iter + 1;
            alph  = alph/2;
            continue
        end
        
        newLA = cost(newXset,newUset,Rset,Xdesset,lambdaSet,mu);
        
        exp = -alph*(delV(1) + alph*delV(2));
        if exp > 0
            z = (LA-newLA)/exp;
        else
            z = -1;
        end
        iter = iter + 1;
        alph  = alph/2;
    end
    
    if newLA > LA
        newXset = newXset*nan(1,1);
        error('cost increased in forward pass')
    end
end

function [newX,newU] = generateTrajectory(Xset,Uset,Kset,dset,Rset,alph,lambdaSet)
    global n m N dt
    newX = nan(n,N);
    newU = nan(m,N-1);
    
    newX(:,1) = Xset(:,1);
    for k = 2:N
        %delrem = newX(1:end-4,k-1) - Xset(1:end-4,k-1);
        %delq = [dot(newX(end-3:end,k-1),Xset(end-3:end,k-1)); Xset(end-3,k-1)*newX(end-2:end,k-1)-newX(end-3,k-1)*Xset(end-2:end,k-1) - cross(Xset(end-2:end,k-1),newX(end-2:end,k-1))];
        %delang = delq(2:end)/(1+delq(1));
        %newU(:,k-1) = Uset(:,k-1) + Kset(:,:,k-1)*[delrem;delang] + alph*dset(:,k-1);
        delx = Gmat(Xset(end-3:end,k-1))*(newX(:,k-1) - Xset(:,k-1));%*(newX(:,k-1) - [Xset(1:end-4,k-1);zeros(4,1)]); %I am not confident in this. Look at equation 81 in the AL_iLQR Tutorial by Jackson
        delu = Kset(:,:,k-1)*delx + alph*dset(:,k-1);

        newU(:,k-1) = Uset(:,k-1) + delu;
        xk = rk4(newX(:,k-1),newU(:,k-1),Rset(:,dt*(k-1)),(k-1)*dt); 
        xk(end-3:end) = xk(end-3:end)/norm(xk(end-3:end));
        newX(:,k) = xk;
    end
end


function newX = generateTrajectory0(x0,Uset,Rset)
    global n N dt
    newX = nan(n,N);
    
    newX(:,1) = x0;
    for k = 2:N
        %newX(:,k) = rk4(newX(:,k-1),Uset(:,k-1),Rset(:,k-1),k*dt); 
        xk = rk4(newX(:,k-1),Uset(:,k-1),Rset(:,dt*(k-1)),(k-1)*dt); 
        xk(end-3:end) = xk(end-3:end)/norm(xk(end-3:end));
        newX(:,k) = xk;
%         if any(isnan(newX(:,k)))
%             rk4(newX(:,k-1),Uset(:,k-1),Rset(:,dt*(k-1)),(k-1)*dt)
%         end
    end
end

function [] = projection()
    
end

function [] = linesearch()
    alph = 1;
    while true
       dlam = regsolve(S1,d,S2,solTol,iterNum);
       dy = -HinvD*dlam;
       newy = y + dy*alph;
       dyn constraints?
       
    end
end

function x = regsolve(A,b,B,tol,countMax)
    x = B\b;
    c = 0;
    while c < countMax
        r = b-A*x;
        if norm(r) < tol
            break
        else
            x = x + B\r;
            c = c + 1;
        end
    end
end

function [Xset,Uset,lambdaSet,Kset,Pset] = trajOpt(x0,Rset,Xdesset)
    global m N LA0
    Uset = rand(m,N-1)/1000;
    Xset = generateTrajectory0(x0,Uset,Rset);
    nonfinflag = any([any(isnan(Xset),'all'),any(isnan(Uset),'all'),any(isinf(Xset),'all'),any(isinf(Uset),'all')]);
    if nonfinflag
        Xset
        Uset
        error('invalid first control sequence')
    end
    [Xset,Uset,lambdaSet,Kset,Pset,mu] = alilqr(Xset,Uset,Rset,Xdesset);
    LA = cost(Xset,Uset,Rset,Xdesset,lambdaSet,mu)
    LAnc = cost(Xset,Uset,Rset,Xdesset,0*lambdaSet,0*mu)
    LA0 = LA0
    %[Xset,Uset,lambdaSet,Kset] = projection(Xset,Uset,lambdaSet,Kset);
    
end

function [Xset,Uset,lambdaSet,Kset,Pset,mu] = alilqr(Xset,Uset,Rset,Xdesset)
    global lagMultInit numc N penInit regInit maxOuterIter maxIlqrIter gradTol costTol cmax zcountLim maxIter penMax penScale m numIneq lagMultMin lagMultMax ilqrCostTol
    global LA0
    iter = 0;
    %dLA = 1/eps;
    grad = 1/eps;
    %delV = [1/eps,1/eps];
    lambdaSet = lagMultInit*ones(numc,N);
    mu = penInit;
    LA0 = cost(Xset,Uset,Rset,Xdesset,lambdaSet,mu);
    %AL
    for j = 1:maxOuterIter
        j
        cmaxtmp = 0;
        dlaZcount = 0;
        LA = cost(Xset,Uset,Rset,Xdesset,lambdaSet,mu);
        %LA0 = LA;
        iter = iter + 1;
        %ILQR
        rho = regInit;
        drho = regInit;
        for ii = 1:maxIlqrIter
            %rho
            %drho
            [Pset,Kset,dset,delV,rho,drho] = backwardpass(Xset,Uset,Rset,Xdesset,lambdaSet,rho,drho,mu);
            [newXset,newUset,newLA,rho,drho] =  forwardpass(Xset,Uset,Kset,Rset,dset,delV,LA,lambdaSet,rho,drho,mu,Xdesset);
            grad = mean(max(abs(dset)./(abs(Uset)+1),[],1));
            iter = iter + 1;
            dLA = abs(newLA-LA)
            dlaZcount = (dLA == 0)*(dlaZcount + (dLA == 0));
            clist = nan(numc,N);
            for k = 1:N-1
                lamk = lambdaSet(:,k);
                ck = constraints(k,newXset(:,k),newUset(:,k));
                Imuk = Imu(mu,ck,lamk);
                clist(:,k) = (Imuk>0)*ck;
            end
            lamk = lambdaSet(:,N);
            ck = constraints(N,newXset(:,N),zeros(m,1));
            Imuk = Imu(mu,ck,lamk);
            clist(:,N) = (Imuk>0)*ck;
            cmaxtmp = max(max(abs(clist)))
%             
%             if any(any(lambdaSet~=0))
%                 figure;
% %                 subplot(2,1,1)
%                 plot([-1 ,1],[0.2,0.2],'k')
%                 hold on
%                 plot([-1 ,1],[-0.2,-0.2],'k')
%                 plot([0.2,0.2],[-1 ,1],'k')
%                 plot([-0.2,-0.2],[-1 ,1],'k')
%                 tester = or((lambdaSet(1:3,1:end-1)~= 0),(lambdaSet(4:6,1:end-1)~= 0));
% %                 scatter(Uset(1,:).*tester(1,:),newUset(1,:).*tester(1,:),'b')
% %                 scatter(Uset(2,:).*tester(2,:),newUset(2,:).*tester(2,:),'b')
% %                 scatter(Uset(3,:).*tester(3,:),newUset(3,:).*tester(3,:),'b')
%                 scatter(Uset(1,:),newUset(1,:),'b')
%                 scatter(Uset(2,:),newUset(2,:),'b')
%                 scatter(Uset(3,:),newUset(3,:),'b')
% %                 scatter(Uset(1,:),newUset(1,:)-Uset(1,:),'g+')
% %                 scatter(Uset(2,:),newUset(2,:)-Uset(2,:),'g+')
% %                 scatter(Uset(3,:),newUset(3,:)-Uset(3,:),'g+')
% %                 scatter(Uset(1,:),dset(1,:),'r^')
% %                 scatter(Uset(2,:),dset(2,:),'r^')
% %                 scatter(Uset(3,:),dset(3,:),'r^')
% %                 subplot(2,1,2)
% %                 scatter(newUset(1,:)-Uset(1,:),dset(1,:),'r^')
% %                 hold on
% %                 scatter(newUset(2,:)-Uset(2,:),dset(2,:),'r^')
% %                 scatter(newUset(3,:)-Uset(3,:),dset(3,:),'r^')
%                 drawnow
%             end
            
            
            Xset = newXset;
            Uset = newUset;
            LA = newLA;
%             if (((cmaxtmp<cmax || j < maxOuterIter) && grad<gradTol && 0 < dLA && dLA < ilqrCostTol) ||...
%                     dlaZcount > zcountLim )
            if (((cmaxtmp<cmax || j < maxOuterIter) && grad<gradTol) ||...
                    dlaZcount > zcountLim ||...
                    (0 < dLA && dLA < ilqrCostTol && (cmaxtmp<cmax || j < maxOuterIter ) ))
%             if (dlaZcount > zcountLim ||...
%                     (0 <= dLA && dLA < ilqrCostTol && grad < gradTol ))
                break
            end
            
            if iter > maxIter
                error('Total iteration limit exceeded')
            end
            
            
        end
        % end of inner loop
            
        %LA = cost(Xset,Uset,Rset,Xdesset,lambdaSet,mu);
%         if (cmaxtmp<cmax || mu >= penMax)
        if (cmaxtmp<cmax && ( mu >= penMax || (0 <= dLA && dLA < costTol) || grad <= gradTol))
%         if (cmaxtmp<cmax && ((0 <= dLA && dLA < costTol) || grad <= gradTol))
            break
        end
        
        for k = 1:N
            for i = 1:numc
                lambdaSet(i,k) = max(-lagMultMax,min(lagMultMax,lambdaSet(i,k) + mu*clist(i,k)));
                if i <= numIneq
                    lambdaSet(i,k) = max(0,lambdaSet(i,k));
                end
            end
        end
        mu = max(0,min(penMax, penScale*mu));
    end
end

function [Kset2,Sset] = TVLQRconst(Xset,Rset)
    global N m n QN  dt numDOF R
    Kset2 = nan(m,numDOF,N);
    Sset = nan(numDOF,numDOF,N);
    
    k=N;
    Sk = QN;
    Sset(:,:,k) = Sk;
  
    qk = Xset(end-3:end,k);
    Gk = Gmat(qk);
    
    for k = N-1:1
        Gkp1 = Gk;
        tk = k*dt;
        rk = Rset(:,tk);
        xk = Xset(:,k);
        qk = xk(end-3:end);
        Skp1 = Sk;
        
        Gk = Gmat(qk);
        [Ak,Bk] = rk4Jacobians(xk,uk,rk,tk);
        Aqk = Gkp1*Ak*Gk.';
        Bqk = Gkp1*Bk;
        
        Kk = (R+Bqk.'*Skp1*Bqk)\Bqk.'*Skp1*Aqk;
        Kset2(:,:,k) = Kk;
        
        Sk = Q(k,rk) + Kk.'*R*Kk + (Aqk-Bqk*Kk)\Skp1*(Aqk-Bqk*Kk);
        Sk = 0.5*(Sk+Sk.');
        Sset(:,:,k) = Sk;
    end
end