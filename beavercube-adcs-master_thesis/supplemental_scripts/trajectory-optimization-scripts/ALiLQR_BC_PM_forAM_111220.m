%Matlab ALILQR
%Patrick McKeen October 2020

%% parameters for run

%n is number of elements in state, m is number of elements in control, N is
%number of time steps. quaternions should always be last 4 elements of
%state
global N dt freq Nslew
N = 995;
Nslew = 0.5;%round(N/3); %if you want to test the satellite slewing to a trajectory and then maintaining that trajectory, set this to some value between 1 and N, indicating when you want the slew to be done. If you want to not use a slew, set this to 0.5.
dt = 1; %s
freq = 1;% How often (in the *dt* time step), do we have a point we care about. If every point on the trajectory matters, use freq = 1. However, if you use freq <1, than only every 1/freq integration point will be primarily considered. This allows the algorithm to use the in-between points to correct itself (such as by slewing out of plane, correcting, and then coming back into it). Freq > 1 should not be used.

% potentially useful prepared vals
global umax
uxMax = 0.15;%0.15; %Am^2 %this is set to less than 0.2 because if we fly a pplanned trajectory that requires 0.2, and we need to adjust using the TVLQR due to disturbances, we won't be able to.
uyMax = 0.15;%0.15; %Am^2
uzMax = 0.15;%0.15; %Am^2
umax = [uxMax;uyMax;uzMax];

global useCost2 usePresetU useSQRT useGuess useAngSquared useExpAngle useGuessOmega useGraph
useCost2 = true; %use the 2D angle cost--this is how closesly a rotated vector aligns with another vector (and thus ignores rotation about that vector). Set to false if you want to fully control 3D orientation
usePresetU = false; %use the same initial U vector as the previous run (if false, a new one is randomly generated). This should be true if you want to control comparisions between different runs
useTrajGoal = true; %if false, only goal is an endpoint. If true, follows an overall trajectory
useRandomInit = false; %if false, assumes it is on correct trajectory at starting point--this is only for orientation! The angular velocity at start is still random.
useDesAsGuess = false; %use the "desired" trajectory as the starting point? Alternative is a guess which is defined later in this code. Probably should be kept as true.
reInitialize = true; %generate new trajectory, starting time (in orbit), orientation, angular velocity, COM, etc. Should be false if you want to re-run repeated trials on same scenario with changes to other settings
useSQRT = false; %use sqrt backpass
useGuess = true; %can give trajGuess--otherwise starts with a random trajectory
useAngSquared = false; %only relevant if UseCost2 is true. Instead of using the angle between the rotated vector and the desired rotated vector, use its square. This can help make the somewhat-small range of angles have more weight and converge easier.
useExpAngle = false; %only relevant if UseCost2 is true. Instead of using the angle between the rotated vector and the desired rotated vector, use exp(rad2deg(angle)). This can help make the somewhat-small range of angles have more weight and converge easier.. takes precedence over the squared angle (if both are true, this one is used)
useGuessOmega = true; %set an omega for the desired trajectory. If true, it generates a suggested velocity profile based on its desired orienatation. If false, the desired angular velocity is assumed to be zeros, so the cost function weights against higher velocity.
useGraph = true; %show graphic response



global satAlignVector1 satAlignVector2 ECIAlignVector1 ECIAlignVector2
satAlignVector1 = [0 0 -1].';
satAlignVector2 = [1 0 0].';
ECIAlignVector1 = @(k,rk,vk) -vk./vecnorm(vk);
ECIAlignVector2 = @(k,rk,vk) -rk./vecnorm(rk);

global swslew swpoint sv1 sv2 su sratioslew sslack
swslew = 0.0001;%0.0001;%10/max(abs(wdes)+0.01,[],'all')^2; %importance of angular velocity during slew or off-freq
swpoint = 1e2*(rad2deg(1))^2;%100/max(abs(wdes)+0.01,[],'all')^2;%importance of angular velocity during regular times
sv1 = 1e4;%1e8; %importance of aoreitnation or angular difference
sv2 = 1e3;
su = 1e3;%1/max(abs(mdes)+1e-6,[],'all'); %importance of u
sratioslew = 0.001;%0.0001;%ratio of importance of params (except ang vel) during slew or off-freq
sslack = 1e14;%max([swslew swpoint sv1 sv2 su sratioslew]); %importance of slacks in cost %don't actualyl want slacks to be too important to cost--they will be removed by constraining them to 0, and by allowing their cost to be comparatively small, a more ideal trajectory is found initially.
 
global gravitygradient_on aero_on srp_on dipole_on prop_on
%Status of different torques (on = 1)
gravitygradient_on = 0;
aero_on = 0;
srp_on = 0;
dipole_on = 0;
prop_on = 0;

rcondcutoff = 25;
global Bset %this is set below.


%% algorithm parameters
%shared params
global maxIter costTol
costTol = 1e-4;
maxIter = 700;

%ilqr params
global gradTol maxIlqrIter beta1 beta2 maxLsIter regInit regMax regMin regScale costMax regBump zcountLim ilqrCostTol maxControl maxState
gradTol = 1e-5;
ilqrCostTol = 10*costTol;
maxIlqrIter = 35; %250
beta1 = 1e-8;%1e-8;
beta2 = 10;
maxLsIter = 10;
regInit = 0;
regMax = 1e8;
regMin = 1e-8;
regBump = 10000000;
regScale = 1.6;%2;%5;%1.6;
costMax = 1e8;
zcountLim = 1;
maxControl = 1e8; %this is NOT the constraint value. This is a number to throw an error if we see it
maxState = 1e8;
global randnoise
randnoise = 0.00;

%AL params
global cmax maxOuterIter penMax penScale penInit lagMultMax lagMultMin lagMultInit dJcountLim slackmax
cmax = 1e-3;
slackmax = 1e-8;
maxOuterIter = 15;
dJcountLim = 10;
penMax = 1e18;%1e8;
penScale = 50;
penInit = 100;
lagMultMax = 1e10; %1e8;
%lagMultMin = -1e4;%-1e8;
lagMultInit = 0;

%projection params
global projLSRegTol projLSRegIter projLSIter projMaxRef projConvRateThreshold projRegPrimal projRegChol projMaxIter projcmax
projLSRegTol = 1e-8;
projLSRegIter = 25;
projLSIter = 10;
projMaxRef = 10;
projConvRateThreshold = 1.1;
projRegPrimal = 1e-8;
projRegChol = 1e-2;
projMaxIter = 2;
% projcmax = 0.5*cmax;


%% replacing  simulation_initialize_PM (when we run this in the simulation, this portion should be removed or modified or both)
if reInitialize
    sim_length = 1000; %~1 ISS orbit

    load('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/r_ECI.mat');
    load('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/v_ECI.mat');
    load('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/B_ECI.mat');
    load('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/sun_ECI.mat');
    B_ECI = B_ECI(1:sim_length, :);
    r_ECI = r_ECI(1:sim_length, :);
    v_ECI = v_ECI(1:sim_length, :);
    sun_ECI = sun_ECI(1:sim_length, :);
    
    %DEFINE COMMAND MODULE CONSTANT PARAMETERS%
    qf_desired = zeros(sim_length,4);
    qf_guess = zeros(sim_length,4);
    rotlim = 0.5;
    for i=1:sim_length
        q_command = qcommand_PM(satAlignVector1, satAlignVector2, ECIAlignVector1(i,r_ECI(i, :).',v_ECI(i, :).'), ECIAlignVector2(i,r_ECI(i, :).',v_ECI(i, :).'));
        q_gc = qcommand_PM(satAlignVector1, (R_z(360*i/300)*satAlignVector2),ECIAlignVector1(i,r_ECI(i, :).',v_ECI(i, :).'), ECIAlignVector2(i,r_ECI(i, :).',v_ECI(i, :).'));
        if i > 1
            q_prev = qf_desired(i-1,:);
            sign_prev = sign(q_prev);
            sign_current = sign(q_command);
            if abs(q_prev(1) - q_command(1)) > rotlim || abs(q_prev(2)-q_command(2)) > rotlim || abs(q_prev(3)-q_command(3)) > rotlim || abs(q_prev(4)-q_command(4)) > rotlim
                q_command = q_command*-1;
            end


            q_gp = qf_guess(i-1,:);
            sign_prev_g = sign(q_gp);
            sign_current_g = sign(q_gc);
            if abs(q_gp(1) - q_gc(1)) > rotlim || abs(q_gp(2)-q_gc(2)) > rotlim || abs(q_gp(3)-q_gc(3)) > rotlim || abs(q_gp(4)-q_gc(4)) > rotlim
                q_gc = q_gc*-1;
            end
        end
        qf_desired(i, :) = q_command;
        qf_guess(i, :) = q_gc;
    end
    qf_desired(end-3:end,:) = qf_desired(end-3:end,:)./vecnorm(qf_desired(end-3:end,:));
    qf_guess(end-3:end,:) = qf_guess(end-3:end,:)./vecnorm(qf_guess(end-3:end,:));


    %Disturbance torques
    global prop_axis prop_mag
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

    
    global x0
    w_mag_mean = deg2rad(0.5);
    w_mag_var = deg2rad(0.1);
    w_bias_mean = 0;
    w_bias_var = deg2rad(5);
    x_init = [generate_q(); generate_w(w_mag_mean, w_mag_var)]; %w in rad/s
    x0 = [x_init(5:7);x_init(1:4)];
    x0(end-3:end) = x0(end-3:end)./norm(x0(end-3:end));
    
    orbit0 = randi(max(1,round(0.9*size(r_ECI,1)-N*dt-5)),1);
end




%%
Rset = r_ECI(orbit0:end,:).'; 
Vset = v_ECI(orbit0:end,:).';
Bset = B_ECI(orbit0:end,:).';
qdes = qf_desired(orbit0:end,:).';
qguess = qf_guess(orbit0:end,:).';
% Vset = repmat([sind(30);0;cosd(30)],[1,length(v_ECI)]);
% size(Vset)
global J
J = [31364908.06, -6713613.57, 58830.40;
    -6713613.57,10040919.97,-123347.56;
    58830.40,-123347.56,34091278.27]*10^-9; %kg*m^2

% % x_init = [-0.08 -0.1  -0.7 0.71 0.003 0.007 0.006].';
% x0 = [0;0;0;1;0;0;0];
[B_gram,B_cond] = magnetic_grammian(Bset);

%GLOBAL VARIABLES ARE USED FOR PARAMETERS/CONSTANTS/ETC THAT DO NOT CHANGE
%DURING RUNTIME.
%there is basically no checking for consistent dimensions.
global QN R 
% global U0
global LA0 cmax0 slackcmax slackLA slackLAnc LAdone LAdonenc cmaxdone LAnc0
LA0 = 1/eps;
cmax0 = 1/eps;
slackcmax = 1/eps;
slackLA = 1/eps;
slackLAnc = 1/eps;
LAdone = 1/eps;
LAdonenc = 1/eps;
cmaxdone = 1/eps;
LAnc0 = 1/eps;


tcut = find(B_cond < rcondcutoff,1,'first')
if  tcut > Nslew
    warning("Nslew too short.")
end

global rNslew vNslew
rNslew = Rset(:,dt*(ceil(Nslew)-1)+1);
vNslew = Vset(:,dt*(ceil(Nslew)-1)+1);

qdes = qdes(:,dt*((1:N+2) - 1)+1);
wdes = zeros(3,N);
wguess = zeros(3,N);

for k = 2:(N)+1

    qk = qdes(:,k);
    qkm1 = qdes(:,k-1);
    
    dq = [dot(qk,qkm1);-qk(1)*qkm1(2:end)+qkm1(1)*qk(2:end)+cross(qk(2:end),qkm1(2:end))];
%     dq = quatdivide(qk.',qkm1.');
    ang = 2*acos(dq(1));
    if ang > pi
        ang = ang - 2*pi;
    end
    if ang == 0 || all(dq(2:end) == 0)
        vec = zeros(3,1);
        ang = 0;
%         dq(2:end) = 0;
    else
        vec = dq(2:end)/sin(ang/2);%vec = dq(2:end)/sqrt(1-dq(1)^2);
    end
    
    
    wdes(:,k-1) = vec*ang/dt;

    qk = qguess(:,k);
    qkm1 = qguess(:,k-1);
    dq = [dot(qk,qkm1);-qk(1)*qkm1(2:end)+qkm1(1)*qk(2:end)+cross(qk(2:end),qkm1(2:end))];
%     dq = quatdivide(qk.',qkm1.');
    ang = 2*acos(dq(1));
    if ang > pi
        ang = ang - 2*pi;
    end
    if ang == 0 || all(dq(2:end) == 0)
        vec = zeros(3,1);
        ang = 0;
%         dq(2:end) = 0;
    else
        vec = dq(2:end)/sin(ang/2);%vec = dq(2:end)/sqrt(1-dq(1)^2);
    end
    
    wguess(:,k-1) = vec*ang/dt;
end
qdes = qdes(:,1:N);
if useGuessOmega
    wdes = wdes(:,1:N);
else
    wdes = zeros(3,N);
end
xdes = [wdes;qdes];
xdes(end-3:end,:) =  xdes(end-3:end,:)./vecnorm(xdes(end-3:end,:));

qguess = qguess(:,1:N);
wguess = wguess(:,1:N);
xguess = [wguess;qguess];
xguess(end-3:end,:) =  xguess(end-3:end,:)./vecnorm(xguess(end-3:end,:));


xNslew = xdes(:,ceil(Nslew));


QN = Q(N,Rset(:,dt*(N-1)+1),Vset(:,dt*(N-1)+1)); %assuming quaternions are part of state, this should be N-1 x N-1, symmetric, real, and positive semi-definite
% QN = [swpoint*eye(3,3) zeros(3,3);zeros(3,3) sv2*eye(3,3)];
R = su*eye(3,3);


if useTrajGoal
    xf = [];%xdes(:,end);
    trajDes = [repmat(xNslew,[1 ceil(Nslew)]) xdes(:,ceil(Nslew)+1:end)]; %should be empty or n x (N)
    trajDes(end-3:end,:) =  trajDes(end-3:end,:)./vecnorm(trajDes(end-3:end,:));
else
    trajDes =[];
    xf = xdes(:,end);
    xf(end-3:end) = xf(end-3:end)./norm(xf(end-3:end));
end



if ~useRandomInit
    x0 = xdes(:,1);
end

trajGuess = [];
if useGuess
    if useDesAsGuess
        stateGuess = xdes;
    else
        stateGuess = xguess;
    end
    
    trajGuess0 = stateGuess;
    
    q1 = x0(end-3:end);
    q2 = stateGuess(end-3:end,ceil(Nslew));
    w1 = x0(1:3)*max(ceil(Nslew)-1,1)*dt;
    w2 = stateGuess(1:3,ceil(Nslew))*max(ceil(Nslew)-1,1)*dt;
    tt = linspace(0,1,ceil(Nslew));
    [qg,wg] = quatAngVelInterp(tt,q1,q2,w1,w2);
    wg = wg/(max(ceil(Nslew)-1,1)*dt);
    trajGuess0(1:3,1:ceil(Nslew)) = wg;
    trajGuess0(4:7,1:ceil(Nslew)) = qg;
    
    trajGuess = trajGuess0;
    trajGuess(end-3:end,:) =  trajGuess(end-3:end,:)./vecnorm(trajGuess(end-3:end,:));
    
    
    trajGuess(:,1) = x0;
end
% trajGuess = xdes;
% trajGuess(:,1) = x0;
% trajGuess02(:,1) = x0;
% x0 = trajGuess(:,1);
% trajGuess = trajDes
% figure; plot(trajGuess.')

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


%% setup
%Xset = load('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/Xtraj_test.mat').xtraj;
Xset = load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/generateInitialTrajectoryTest.mat').newX;
%Xset = load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/generateTrajectoryTest.mat').Xset;
Uset = load('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/Utestcpp.mat').Utestcpp;
Xdesset = load('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/Xdesset_backwardspass.mat').Xdesset_backwardspass;
lambdaSet = zeros(6, 1000);
slackset = zeros(7, 1000);
useSlacks = 0;
cutoffInt = 1000;
Rset = r_ECI.';
Vset = v_ECI.';
cutoffInt = 3;
%N = cutoffInt;
N = 995;
Xset = Xset(:, 1:N);
Uset = Uset(:, 1:N-1);
Rset = Rset(:, 1:N);
Vset = Vset(:, 1:N);
Xdesset = Xdesset(:, 1:N);
lambdaSet = lambdaSet(:, 1:N);
%[Kset2,Sset] = TVLQRconst(Xset,Uset,Rset,Vset);

% LA = cost2(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,40,slackset,useSlacks)
%k = N;
xk = Xset(:, N);
uk = Uset(:, N-1);
xdesk = Xdesset(:, N);
rk = Rset(:, N);
vk = Vset(:, N);
sk = slackset(:, N);
Xdesset = Xdesset(:, 1:N);
Xdesset = load('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/Xdesset_backwardspass.mat').Xdesset_backwardspass;
Xdesset = Xdesset(:, 1:N);
%Xdesset = setup(x0,trajDes,xf,trajGuess);
%[Kset2,Sset] = TVLQRconst(Xset,Uset,Rset,Vset);
%% run the code
useGuess = 0;
useGuessOmega = 0;
tic
[Kset2,Sset] = TVLQRconst(Xset,Uset,Rset,Vset);
% xk = [-0.2; 0.5; 0.01; 0.1826; 0.3651; 0.5477; 0.7303];
% xk = [-0.2; 0.5; 0.01; 0.1104; 0.4417; 0.7730; -0.4417];
% uk = [0.002; 0.005; 0.001];
% 
% [Ak,Bk] = rk4Jacobians(xk,uk,rk,1)
% [newX,newslack] = generateTrajectory0(xk,Uset,Rset,Vset,trajGuess,useSlacks,dt,N)
% rho = 0;
% drho = 0;
% mu = 40;
%[Pset,Kset,dset,delV,rho,drho, Qkx_debug_matlab, Qku_debug_matlab] = backwardpass(newX,Uset,Rset,Vset,Xdesset,lambdaSet,rho,drho,mu,slackset,useSlacks)
%alph = 0.01;
%LA = cost2(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,mu,slackset,useSlacks);
%[newXset,newUset,newLA,rho,drho,newslackset] = forwardpass(Xset,Uset,Kset,Rset,Vset,dset,delV,LA,lambdaSet,rho,drho,mu,Xdesset,slackset,useSlacks);
%N = 1000;
%[Kset2,Sset] = TVLQRconst(Xset,Uset,Rset,Vset);
%load('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/testALILQR2.mat');
%[Pset,Kset,dset,delV,rho,drho, Qkx_debug_matlab, Qku_debug_matlab] = backwardpass(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,rho,drho,mu,slackset,useSlacks);
%[Xset_out,Uset_out,lambdaSet,Kset,Pset,mu,rho,drho,slackset] = alilqr(Xset,Uset,Rset,Vset,Xdesset,slackset,useSlacks,false, [], []);
warning(2);
%[Pset,Kset,dset,delV,rho,drho, Qkx_debug_matlab, Qku_debug_matlab] = backwardpass(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,rho,drho,mu,slackset,useSlacks);
%LA = cost2(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,mu,slackset,useSlacks);
%save('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/Kgentraj.mat', 'Kset');
%save('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/dgentraj.mat', 'dset');
%alph = 0.01;
%N = 50;
%[newXset,newUset,newslackset] = generateTrajectory(Xset,Uset,Kset,dset,Rset,Vset,alph,lambdaSet,slackset,useSlacks);
%[newXset,newUset,newLA,rho,drho,newslackset] = forwardpass(Xset,Uset,Kset,Rset,Vset,dset,delV,LA,lambdaSet,rho,drho,mu,Xdesset,slackset,useSlacks);
%warning(2);
%system_string = './Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/TrajectoryPlanning/root/MatrixTest/MatrixTest ';
%system('cd /Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/TrajectoryPlanning/root/MatrixTest && ./MatrixTest');
%system('./MatrixTest');
%Qkx_debug_cpp = load('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/testBpassQkx.mat').Qkx_out;
%Qku_debug_cpp = load('/Users/alexmeredith/starlab-beavercube-adcs/beavercube-adcs/matfiles/testBpassQku.mat').Qku_out;
%Qkx_err = zeros(1,N);
% for errind=1:N
%     Qkx_err(errind) = sum((Qkx_debug_cpp(:, errind)-Qkx_debug_matlab(:, errind)).^2);
% end
% plot(Qkx_err)
% warning(2)
[Xset,Uset,lambdaSet,Kset,Pset,slackX,slackU] = trajOpt(x0,Rset,Vset,Xdesset,trajGuess);
timetook = toc
[Kset2,Sset] = TVLQRconst(Xset,Uset,Rset,Vset);
vv = zeros(3,N);
uu = zeros(3,N);
philist = zeros(1,N);
q0klist = zeros(4,N);
for k = 1:N
    [v,u] = pvs(k,Rset(:,dt*(k-1)+1),Vset(:,dt*(k-1)+1));
    vv(:,k) = v;
    uu(:,k) = u;
    qk = Xset(end-3:end,k);
    [phi,~,q0k] = cost2ang(qk,v,u);
    q0klist(:,k) = q0k;
    philist(k) = rad2deg(phi);
end
angdiff0 = rad2deg((-Xdesset(5:7,:).*(ones(3,1)*Xset(4,:))...
    +Xset(5:7,:).*(ones(3,1)*Xdesset(4,:))...
    -cross(Xdesset(5:7,:),Xset(5:7,:)))...
    ./(1 + sum(Xset(4:7,:).*Xdesset(4:7,:),1)));
rotvec = quatrotate(quatconj(Xset(4:7,:).'),vv.').';
rotvec = rotvec./vecnorm(rotvec);
angdiff = real(rad2deg(acos(dot(rotvec,uu))));
% figure; plot((Xset(1:3,:)-Xdesset(1:3,:)).')
% title('omega diff')
% figure; plot((Xset(4:7,:)-Xdesset(4:7,:)).')
% title('quat diff')
figure; plot((Xset(1:3,:)).')
title('omega')
figure; plot((Xset(4:7,:)).')
title('quat')
figure; plot(Uset.')
title('u')
figure; plot(angdiff.')
title('ang between rotated vector and desired rotated vector')
figure; plot(rotvec.')
title('rotated vector')
% figure; plot(angdiff0.')
% title('linearized angle vector between "desired" and actual quat')
figure; semilogy(B_cond)
title('Magnetic Grammian Condition Number')
tcut

% figure; plot(q0klist.')
% title('closest desired quaternion that meets reqs')

% figure;
% plot(philist)
% title('ang between rotated vector and desired rotated vector')

superUset1 = interp1(dt*((1:N-1)-1)+1,Uset.',1:(N*dt)).';

[superXset1,~] = generateTrajectory0(x0,superUset1,Rset,Vset,[],false,1,N*dt);


superUset2 = interp1(dt*((1:N-1)-1)+1,Uset.',1:(N*dt),'previous').';

[superXset2,~] = generateTrajectory0(x0,superUset2,Rset,Vset,[],false,1,N*dt);

supervv1 = zeros(3,N);
superuu1 = zeros(3,N);
supervv2 = zeros(3,N);
superuu2 = zeros(3,N);
superphilist1 = zeros(1,N*dt);
superq0klist1 = zeros(4,N*dt);
superphilist2 = zeros(1,N*dt);
superq0klist2 = zeros(4,N*dt);
for k = 1:(N*dt)
    [v,u] = pvs(k/dt,Rset(:,k),Vset(:,k));
    qk1 = superXset1(end-3:end,k);
    [phi1,~,q0k1] = cost2ang(qk1,v,u);
    superq0klist1(:,k) = q0k1;
    superphilist1(k) = rad2deg(phi1);
    supervv1(:,k) = v;
    superuu1(:,k) = u;
    supervv2(:,k) = v;
    superuu2(:,k) = u;
    
    
    qk2 = superXset2(end-3:end,k);
    [phi2,~,q0k2] = cost2ang(qk2,v,u);
    superq0klist2(:,k) = q0k2;
    superphilist2(k) = rad2deg(phi2);
end



superrotvec1 = quatrotate(quatconj(superXset1(4:7,:).'),supervv1.').';
superrotvec1 = superrotvec1./vecnorm(superrotvec1);
superangdiff1 = real(rad2deg(acos(dot(superrotvec1,superuu1))));
superrotvec2 = quatrotate(quatconj(superXset2(4:7,:).'),supervv2.').';
superrotvec2 = superrotvec2./vecnorm(superrotvec2);
superangdiff2 = real(rad2deg(acos(dot(superrotvec2,superuu2))));
% figure; plot(superUset1.')
% title('superu1')
% figure; plot(superangdiff1.')
% title('super1 ang between rotated vector and desired rotated vector')
% figure; plot(superrotvec1.')
% title('super1 rotated vector')
% figure; plot(superq0klist1.')
% title('super1 closest desired quaternion that meets reqs')


figure; plot(superUset2.')
title('superu2')
figure; plot(superangdiff2.')
title('super2 ang between rotated vector and desired rotated vector')
figure; plot(superrotvec2.')
title('super2 rotated vector')
% figure; plot(superq0klist2.')
% title('super2 closest desired quaternion that meets reqs')


X = Xset; U = Uset; K = Kset;
controller_status = 2;
warning('REMEMBER TO SAVE THIS DATA, PATRICK')

% [XsetB,UsetB,lambdaSetB,KsetB,PsetB,muB,rhoB,drhoB,slacksetB] = alilqr(Xset,Uset,Rset,Vset,Xdesset,[],false,true,lambdaSet,mu*0);


%% functions that define dynamics, metrics, constraints, etc. (These would change between BC and other satellites or different trajectory goals, disturbance torques, etc.)
function Qk = Q(k,rk,vk)
    %assuming quaternions are part of state, this should be N-1 x N-1, symmetric, real, and positive semi-definite
    %this is nadir pointing for now
    global Nslew sv1 swpoint sv2 rNslew vNslew swslew sratioslew dt freq
    if k >= Nslew
%         a = [0 0 0].';
%         b = rk;
%         vk
%         rk
%         e1 = vk;
%         e1 = e1/norm(e1);
%         e2 = rk;%[1;0;0];
%         if abs(dot(e2,e1)) > 0.75
%             e2 = [0;1;0];
%         end
%         e2 = e2-dot(e2,e1)*e1;
%         e2 = e2/norm(e2);
%         e3 = cross(e1,e2);
%         e3 = e3/norm(e3);
%         Qkqq = sv1*(e2*e2.') + sv2*(e3*e3.');
        Qkqq = sv1*eye(3,3);
        Qkqw = zeros(3,3);
        Qkww = swpoint*eye(3,3);
        Qk = [Qkww Qkqw; Qkqw.' Qkqq]; 
        if rem(k,round(1/freq)) ~= 0
            Qk = Qk*sratioslew;
        end
    else
        Qk = Q(Nslew,rNslew,vNslew)*sratioslew;
        Qk(1:3,1:3) = eye(3,3)*swslew;
    end
%      Qk = [0*eye(3,3) zeros(3,3);zeros(3,3) sv1*eye(3,3)];
end

function [v,u] = pvs(k,rk,vk)
    global Nslew rNslew vNslew satAlignVector1 ECIAlignVector1
    
%     v = [0;0;1];
    v = satAlignVector1;
%     u = -rk/norm(rk);
    if k >= Nslew
        u = ECIAlignVector1(k,rk,vk);
        %utest = -vk/norm(vk)
        %vk
        %vkmag = norm(vk)
%         u = -vk/norm(vk);
    else
        u = ECIAlignVector1(Nslew,rNslew,vNslew);
%         u = -vNslew/norm(vNslew);
    end
end

function xdot = dynamics(x,u,r,t) %continuous dynamics
    global J
    w = x(1:3);
    q = x(end-3:end);
    B = magneticFieldECI(r,t);
    dtau = distTorq(r,x,t);
    wdot = -J\(cross(w,J*w) + cross(rotT(q)*B,u) - dtau);
%     wdot = -J\(cross(w,J*w) + u/1000 - dtau);
    qdot = 0.5*Wmat(q)*w;
    xdot = [wdot;qdot];
end

function [jacX,jacU] = dynamicsJacobians(x,u,r,t) %these are the Jacobians of the continuous function. %assuming r (and thus B) is constant over timestep
    global J
    w = x(1:3);
    q = x(end-3:end);
    [Rt1, Rt2, Rt3, Rt4] = Rti(q);
    B = magneticFieldECI(r,t);
    
    [distTorqJacW,distTorqJacQ] = distTorqJac(r,x,t);
    jww = J\(skewsym(J*w) - skewsym(w)*J + distTorqJacW);
    jwq = J\(skewsym(u)*[Rt1*B Rt2*B Rt3*B Rt4*B]+ distTorqJacQ);
%     jwq = J\(distTorqJacQ);
    jqw =  0.5*Wmat(q);
    jqq = 0.5*[0 -w.';w -skewsym(w)];
    jacX = [jww jwq; jqw jqq];
    
    jwu = -J\skewsym(rotT(q)*B);
%     jwu = -J\eye(3,3)/1000;
    jqu = zeros(4,3);
    jacU = [jwu;jqu];
end


%%disturbance models
function pt = propTorq()
    global prop_axis prop_mag
    pt = prop_axis.'*prop_mag;
%     pt = [0;0;1e-6];
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

function dtau = distTorq(r,x,t)
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

function [ jacw,jacq ] = distTorqJac(r,x,t)
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

function Xdesset = setup(x0,trajDes,xf,trajGuess)
    global numc numcp numIneq m n N R QN numDOF useGuess Rs m0 sslack useGraph figlabel
    m = length(R);
    
    if length(unique([size(QN) size(Q(1,0,0))])) > 1
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
        N = size(trajDes,2);
    else
        error('need some sort of goal--specify xf or trajDes')
    end
    
    numc = size(constraints(N,x0,zeros(m,1),zeros(n,1),useGuess),1);
    numIneq = size(ineqConstraints(N,x0,zeros(1,m)),1);
    
    if useGraph
        figlabel = figure;
    end
    
    
    if useGuess
        if any(size(trajGuess) ~= size(Xdesset))
            error('Wrong size trajectory guess')
        end
        Rs = sslack*eye(n,n);
        if any(trajGuess(:,1) ~= x0)
            error('Guess trajectory must start at x0')
        end
        numc = numc-n;
        numcp = numc + n;
    end
end


function c = constraints(k,x,u,s,useSlacks)
%     global cmax
    if useSlacks
%         c = [max(ineqConstraints(k,x,u),-cmax);eqConstraints(k,x,u);s];
        c = [ineqConstraints(k,x,u);eqConstraints(k,x,u);s];
    else
%         c = [max(ineqConstraints(k,x,u),-cmax);eqConstraints(k,x,u)];
        c = [ineqConstraints(k,x,u);eqConstraints(k,x,u)];
    end
end

function [cku,ckx,cks] = constraintJacobians(k,x,u,s,useSlacks)
    global numc n m % numIneq cmax
    c = constraints(k,x,u,s,useSlacks);
    [ecku,eckx] = eqConstraintsJacobians(k,x,u);
    [icku,ickx] = ineqConstraintsJacobians(k,x,u);
    cku = [icku;ecku];
    ckx = [ickx;eckx]*Gmat(x(end-3:end)).';
%     cku = [icku.*(c(1:numIneq)>-cmax);ecku];
%     ckx = [ickx.*(c(1:numIneq)>0);eckx]*Gmat(x(end-3:end)).';
    if useSlacks
        cks = [zeros(numc,n);eye(n,n)];
%         cku = [icku.*(c(1:numIneq)>-cmax);ecku;zeros(n,m)];
%         ckx = [ickx.*(c(1:numIneq)>-cmax);eckx;zeros(n,n)]*Gmat(x(end-3:end)).';
        cku = [icku;ecku;zeros(n,m)];
        ckx = [ickx;eckx;zeros(n,n)]*Gmat(x(end-3:end)).';
    else
        cks = [];
    end
end




function [lkxx,lkuu,lkux,lkx,lku,ckx,cku,lkss,lks,cks] = costJacobians(k,xk,uk,xdesk,rk,vk,sk,useSlacks)
    global N R QN Rs m n useCost2
    if useCost2
        [lkxx,lkuu,lkux,lkx,lku,ckx,cku,lkss,lks,cks] = cost2Jacobians(k,xk,uk,xdesk,rk,vk,sk,useSlacks);
    else
        if useSlacks
            [cku,ckx,cks] = constraintJacobians(k,xk,uk,sk,useSlacks);
            if k == N
                lkuu = R*0;
                lku = uk*0;
                lkux = zeros(m,n)*Gmat(xk(end-3:end)).';

                lkx = QN*Gmat(xk(end-3:end))*(xk-xdesk);
%                 lkx = Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).'*QN*Gmat(xk(end-3:end))*(xk-xdesk);
                lkxx = QN;
%                 lkxx = Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).'*QN*(Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).').';

                lkss = Rs*0;
                lks = xk*0;
            elseif k<N && k>=0
                lkuu = R;
                lku = R*uk;
                lkux = zeros(m,n)*Gmat(xk(end-3:end)).';

                lkx = Q(k,rk,vk)*Gmat(xk(end-3:end))*(xk-xdesk);
%                 lkx = Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).'*Q(k,rk,vk)*Gmat(xk(end-3:end))*(xk-xdesk);
                lkxx = Q(k,rk,vk);
%                 lkxx = Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).'*Q(k,rk,vk)*(Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).').';

                lkss = Rs;
                lks = Rs*sk;
            else
                error('invalid time index')
            end
        else
            [cku,ckx,cks] = constraintJacobians(k,xk,uk,sk,useSlacks);
            if k == N
                lkuu = R*0;
                lku = uk*0;
                lkux = zeros(m,n)*Gmat(xk(end-3:end)).';

                lkx = QN*Gmat(xk(end-3:end))*(xk-xdesk);
%                 lkx = Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).'*QN*Gmat(xk(end-3:end))*(xk-xdesk);
                lkxx = QN;
%                 lkxx = Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).'*QN*(Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).').';
            elseif k<N && k>=0
                lkuu = R;
                lku = R*uk;
                lkux = zeros(m,n)*Gmat(xk(end-3:end)).';

                lkx = Q(k,rk,vk)*Gmat(xk(end-3:end))*(xk-xdesk);
%                 lkx = Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).'*Q(k,rk,vk)*Gmat(xk(end-3:end))*(xk-xdesk);
                lkxx = Q(k,rk,vk);
%                 lkxx = Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).'*Q(k,rk,vk)*(Gmat(xk(end-3:end))*Gmat(xdesk(end-3:end)).').';
            else
                error('invalid time index')
            end
            lkss=[];
            lks=[];
        end
    end
end

function [lkxx,lkuu,lkux,lkx,lku,ckx,cku,lkss,lks,cks] = cost2Jacobians(k,xk,uk,xdesk,rk,vk,sk,useSlacks)
    global N R QN Rs m n
    if useSlacks
        [cku,ckx,cks] = constraintJacobians(k,xk,uk,sk,useSlacks);
        if k == N
            lkuu = R*0;
            lku = uk*0;
            lkux = zeros(m,n)*Gmat(xk(end-3:end)).';

            
           
            vsk = Gmat(xk(end-3:end))*(xk-xdesk);
            [v,u] = pvs(k,rk,vk);
            qk = xk(end-3:end);
            [phi,dphi,~] = cost2ang(qk,v,u);
            wlkx = QN(1:3,1:3)*vsk(1:3);
            wlkxx = QN(1:3,1:3);
            qlkx = dphi.'*QN(4,4)*phi;
            qlkxx = dphi.'*QN(4,4)*dphi;
            lkx = [wlkx;qlkx];
            
            lkxx = [wlkxx zeros(3,3); zeros(3,3) qlkxx];
%             lkx = Q(k,rk,vk)*Gmat(xk(end-3:end))*(xk-xdesk);
%             lkxx = Q(k,rk,vk);
            
            
            lkss = Rs*0;
            lks = xk*0;
        elseif k<N && k>=0
            lkuu = R;
            lku = R*uk;
            lkux = zeros(m,n)*Gmat(xk(end-3:end)).';

            

            Qk = Q(k,rk,vk);
            vsk = Gmat(xk(end-3:end))*(xk-xdesk);
            [v,u] = pvs(k,rk,vk);
            qk = xk(end-3:end);
            [phi,dphi,~] = cost2ang(qk,v,u);
            wlkx = Qk(1:3,1:3)*vsk(1:3);
            wlkxx = Qk(1:3,1:3);
            qlkx = dphi.'*Qk(4,4)*phi;
            qlkxx = dphi.'*Qk(4,4)*dphi;
            lkx = [wlkx;qlkx];
            
            lkxx = [wlkxx zeros(3,3); zeros(3,3) qlkxx];
%             lkx = Q(k,rk,vk)*Gmat(xk(end-3:end))*(xk-xdesk);
%             lkxx = Q(k,rk,vk);
            lkss = Rs;
            lks = Rs*sk;
        else
            error('invalid time index')
        end
    else
        [cku,ckx,cks] = constraintJacobians(k,xk,uk,sk,useSlacks);
        if k == N
            lkuu = R*0;
            lku = uk*0;
            lkux = zeros(m,n)*Gmat(xk(end-3:end)).';
            
            vsk = Gmat(xk(end-3:end))*(xk-xdesk);
            [v,u] = pvs(k,rk,vk);
            qk = xk(end-3:end);
            [phi,dphi,~] = cost2ang(qk,v,u);
            wlkx = QN(1:3,1:3)*vsk(1:3);
            wlkxx = QN(1:3,1:3);
            qlkx = dphi.'*QN(4,4)*phi;
            qlkxx = dphi.'*QN(4,4)*dphi;
            lkx = [wlkx;qlkx];
            
            lkxx = [wlkxx zeros(3,3); zeros(3,3) qlkxx];
%             lkx = Q(k,rk,vk)*Gmat(xk(end-3:end))*(xk-xdesk);
%             lkxx = Q(k,rk,vk);
        elseif k<N && k>=0
            lkuu = R;
            lku = R*uk;
            lkux = zeros(m,n)*Gmat(xk(end-3:end)).';
            Qk = Q(k,rk,vk);
            vsk = Gmat(xk(end-3:end))*(xk-xdesk);
            [v,u] = pvs(k,rk,vk);
            qk = xk(end-3:end);
            [phi,dphi,~] = cost2ang(qk,v,u);

            
            wlkx = Qk(1:3,1:3)*vsk(1:3);
            wlkxx = Qk(1:3,1:3);
            qlkx = dphi.'*Qk(4,4)*phi;
            qlkxx = dphi.'*Qk(4,4)*dphi;
            lkx = [wlkx;qlkx];
            
            lkxx = [wlkxx zeros(3,3); zeros(3,3) qlkxx];
%             lkx = Q(k,rk,vk)*Gmat(xk(end-3:end))*(xk-xdesk);
%             lkxx = Q(k,rk,vk);
            
        else
            error('invalid time index')
        end
        lkss=[];
        lks=[];
    end
end

function [ang,dang,q0k] = cost2ang(qk,v,u)
    global useAngSquared useExpAngle
            dvu = dot(v,u);
            cvu = cross(v,u);
            a = [sqrt((1+dvu)/2); cvu/sqrt(2*(1+dvu))];
            b = [0; (u+v)/sqrt(2*(1+dvu))];
            la = qk.'*a;
            lb = qk.'*b;
            mag = sqrt(la^2+lb^2);
            
%             la0 = la;
%             lb = lb*sign(la);
%             la = la*sign(la);
            
            ss = lb/mag;
            cc = la/mag;
%             cc = cc*sign(ss);
%             ss = ss*sign(ss);
            q0k = (cc*a + ss*b);
%             q0ka = (ss*a - cc*b)/mag;
%             dq = (cc*bs.' - ss*as.')/mag^2;
            dd = dot(qk,q0k);
            if abs(1-dd) < 1e-8
                %vectors basically aligned
                ang = 1-dot(qk,q0k)^2;
                dang = (-2*(la*a.' + lb*b.'))*Wmat(qk);
            elseif abs(1+dd) < 1e-8
                %vectors basically opposite (these are equal bc
                %quaternions)
                ang = 1-dot(qk,q0k)^2;
                dang = (-2*(la*a.' + lb*b.'))*Wmat(qk);
            else
                ang = acos(2*dot(qk,q0k)^2-1);
    %             ang = acos(abs(dot(qk,q0k)));
    %             ang = 1-dot(qk,q0k)^2;
    %             dang = ((-2*(la*a.' + lb*b.')/(mag*sqrt(1-mag^2))))*Wmat(qk)/mag;
                dang = ((-2*(la*a.' + lb*b.')/(mag*sqrt(1-mag^2))))*Wmat(qk);%/mag;
    %             dang = -(1/sqrt(1-mag^2))*sign(dot(qk,q0k))*q0k.'*Wmat(qk);
    %             dang = sign(la)*(-2*(la*a.' + lb*b.'))*Wmat(qk)*sqrt(1-mag^2)/mag;
    %             dang = (-2*(la*a.' + lb*b.'))*Wmat(qk)/mag;
    %             dang = (-2*(la*a.' + lb*b.'))*Wmat(qk)/sqrt(1-mag^2);
    %             dang = (-2*(la*a.' + lb*b.'))*Wmat(qk)*sqrt(1-mag^2)/mag;
                ang = rad2deg(ang);
                dang = rad2deg(dang);
                if useAngSquared
                    dang = 2*ang*dang;
                    ang = ang^2;
                end
                
                if useExpAngle
                    ang = exp(abs(ang));
                    dang = exp(abs(ang))*dang*sign(ang);
                end
                    
                        

    %             syms q1 q2 q3 q4 sphi real
    %             sa = sym('sa',[4,1],'real');
    %             sb = sym('sb',[4,1],'real');
    %             sqk = [q1 q2 q3 q4].';
    %             sla = dot(sqk,sa);
    %             slb = dot(sqk,sb);
    %             sy1 = slb;
    %             cy1 = sla;
    % %             sy1 = sy1*sign(sy1);
    % %             cy1 = cy1*sign(sy1);
    %             smag = sqrt(sla^2+slb^2);
    % %             sy1 = sy1;
    % %             cy1 = cy1;
    % % %             sy2 = -dot(sqk,sa);
    % % %             cy2 = dot(sqk,sb);
    % %             smag = sqrt(sla^2+slb^2);
    % %             tyd2 = (-sla + smag)/slb;
    % %             tyd2a = (-sla - smag)/slb;
    % %             sy1 = 2*tyd2/(1+tyd2^2);
    % %             cy1 = (1-tyd2^2)/(1+tyd2^2);
    % %             sy2 = 2*tyd2a/(1+tyd2a^2);
    % %             cy2 = (1-tyd2a^2)/(1+tyd2a^2);
    %             sq01 = (sa*cy1 + sb*sy1)/sqrt(sy1^2+cy1^2);
    % %             sq02 = (sa*cy2 + sb*sy2);%/sqrt(sy2^2+cy2^2);
    %             sp1 = acos(2*dot(sqk,sq01)^2-1);%Wmat(sq01).'*sqk
    % %             sp2 = acos(2*dot(sqk,sq02)^2-1);%Wmat(sq02).'*sqk
    %             sdp1 = jacobian(sp1,sqk)
    % %             sdp2 = jacobian(sp2,sqk)
    % % % %             simplify(sdp1)
    % % % %             simplify(sdp2)
    % % %             
    % % %             sdq = (sla*sb.' - slb*sa.' + ((slb/smag)*(sla*sa.'+ slb*sb.')-sb.'*smag))/(smag^2-sla*smag);
    % % %             sdqa = (sla*sb.' - slb*sa.' - ((slb/smag)*(sla*sa.'+ slb*sb.')-sb.'*smag))/(smag^2+sla*smag);
    % % % %             
    %             gt1 = (-2*(sla*sa.' + slb*sb.')/(smag*sqrt(1-smag^2)));
    % % %             gt2 = 4*dot(sq02,sqk)*dot(sqk,-sy2*sa+cy2*sb)*sdqa/sqrt(1-(2*dot(sqk,sq02)^2-1)^2)
    % 
    % %                 
    % %             simplify(subs(gt1,[q1, q2, q3, q4, sa1, sa1,sa3,sa4 sb1 sb2 sb3 sb4],[qk.' a.' b.']))-
                
            end
end



function B = magneticFieldECI(rECI,t)
    global Bset
    B = Bset(:,t);
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

function LA = cost(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,mu,slackset,useSlacks)
    global N QN R dt Rs n useCost2
    if useCost2
        LA = cost2(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,mu,slackset,useSlacks);
    else
        LA = 0;
        k = N;
        xk = Xset(:,k);
        lamk = lambdaSet(:,k);
        dLA = 0;
        xdesk = Xdesset(:,k);
        uk = zeros(3,1);
        ck = constraints(k,xk,uk,zeros(n,1),useSlacks);
        Imuk = Imu(mu,ck,lamk,useSlacks);
        dLA = dLA + 0.5*(xk-xdesk).'*Gmat(xk(end-3:end)).'*QN*Gmat(xk(end-3:end))*(xk-xdesk);
        dLA = dLA + max(0,(lamk + 0.5*Imuk*ck).'*ck);
        LA = LA + dLA;
        for k = 1:N-1
            uk = Uset(:,k);
            xk = Xset(:,k);
            vk = Vset(:,dt*(k-1)+1);
            rk = Rset(:,dt*(k-1)+1);
            lamk = lambdaSet(:,k);
            xdesk = Xdesset(:,k);
            if useSlacks
                sk = slackset(:,k);
            else
                sk = [];
            end
            ck = constraints(k,xk,uk,sk,useSlacks);
            Imuk = Imu(mu,ck,lamk,useSlacks);
            dLA = 0;
            dLA = dLA + 0.5*(xk-xdesk).'*Gmat(xk(end-3:end)).'*Q(k,rk,vk)*Gmat(xk(end-3:end))*(xk-xdesk);
            dLA = dLA + 0.5*uk.'*R*uk;
            dLA = dLA + max(0,(lamk + 0.5*Imuk*ck).'*ck);
            if useSlacks
                dLA = dLA + 0.5*sk.'*Rs*sk;
            end
            LA = LA + dLA;
        end
    end
end


function LA = cost2(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,mu,slackset,useSlacks)
    global N QN R dt Rs n
    LA = 0;
    k = N;
    xk = Xset(:,k);
    vk = Vset(:,dt*(k-1)+1);
    rk = Rset(:,dt*(k-1)+1);
    lamk = lambdaSet(:,k);
    dLA = 0;
    xdesk = Xdesset(:,k);
    uk = zeros(3,1);
    ck = constraints(k,xk,uk,zeros(n,1),useSlacks);
    Imuk = Imu(mu,ck,lamk,useSlacks);
    qk = xk(end-3:end);
%     dLA = dLA + 0.5*(xk-xdesk).'*Gmat(xk(end-3:end)).'*QN*Gmat(xk(end-3:end))*(xk-xdesk);
    vsk = Gmat(qk)*(xk-xdesk);
    dLA = dLA + 0.5*vsk(1:3).'*QN(1:3,1:3)*vsk(1:3);
    [v,u] = pvs(k,rk,vk);
    qk = xk(end-3:end);
    [phi,~,~] = cost2ang(qk,v,u);
    dLA = dLA + 0.5*phi.'*QN(4,4)*phi;
    dLA = dLA + max(0,(lamk + 0.5*Imuk*ck).'*ck);
    LA = LA + dLA;
    for k = 1:N-1
        uk = Uset(:,k);
        xk = Xset(:,k);
        vk = Vset(:,dt*(k-1)+1);
        rk = Rset(:,dt*(k-1)+1);
        lamk = lambdaSet(:,k);
        xdesk = Xdesset(:,k);
        if useSlacks
            sk = slackset(:,k);
        else
            sk = [];
        end
        ck = constraints(k,xk,uk,sk,useSlacks);
        Imuk = Imu(mu,ck,lamk,useSlacks);
        dLA = 0;
        Qk = Q(k,rk,vk);
        qk = xk(end-3:end);
        vsk = Gmat(qk)*(xk-xdesk);
        dLA = dLA + 0.5*vsk(1:3).'*Qk(1:3,1:3)*vsk(1:3);
        [v,u] = pvs(k,rk,vk);
        [phi,~,~] = cost2ang(qk,v,u);
        dLA = dLA + 0.5*phi.'*Qk(4,4)*phi;
        dLA = dLA + 0.5*uk.'*R*uk;
        dLA = dLA + max(0,(lamk + 0.5*Imuk*ck).'*ck);
        if useSlacks
            dLA = dLA + 0.5*sk.'*Rs*sk;
        end
        LA = LA + dLA;
    end
end

% function Q = Qmat(q)
%     Q = [q(1) q(2:4).'; -q(2:end) q(1)*eye(3,3) - skewsym(q(2:end))];
% end

% function [Q1, Q2, Q3, Q4] = Qi(q)
%     Q1 = eye(4,4);
%     Q2 = [zeros(1,4);zeros(3,1) -skewsym([1;0;0])];
%     Q2(2,1) = 1;
%     Q2(1,2) = -1;
%     Q3 = [zeros(1,4);zeros(3,1) -skewsym([0;1;0])];
%     Q3(3,1) = 1;
%     Q3(1,3) = -1;
%     Q4 = [zeros(1,4);zeros(3,1) -skewsym([0;0;1])];
%     Q4(4,1) = 1;
%     Q4(1,4) = -1;
% end

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
   
    R1 = 2*[q1 q4 -q3; -q4 q1 q2; q3 -q2 q1].';
    R2 = 2*[q2 q3 q4; q3 -q2 q1; q4 -q1 -q2].';
    R3 = 2*[-q3 q2 -q1; q2 q3 q4; q1 q4 -q3].';
    R4 = 2*[-q4 q1 q2; -q1 -q4 q3; q2 q3 q4].';
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
    
    Rt1 = 2*[q1 q4 -q3; -q4 q1 q2; q3 -q2 q1];
    Rt2 = 2*[q2 q3 q4; q3 -q2 q1; q4 -q1 -q2];
    Rt3 = 2*[-q3 q2 -q1; q2 q3 q4; q1 q4 -q3];
    Rt4 = 2*[-q4 q1 q2; -q1 -q4 q3; q2 q3 q4];
end

function vx = skewsym(v)
    vx=[0 -v(3) v(2) ; v(3) 0 -v(1) ; -v(2) v(1) 0 ];
end

function W = Wmat(q)
    W = [-q(2:end).'; q(1)*eye(3,3) + skewsym(q(2:end))];
end

%%%%%%%%%%%%%%%%%
function xkp1 = rk4(xk,uk,rk,tk,dtl)
    k1 = dynamics(xk,uk,rk,tk);
    k2 = dynamics(xk+0.5*dtl*k1,uk,rk,tk);
    k3 = dynamics(xk+0.5*dtl*k2,uk,rk,tk);
    k4 = dynamics(xk+dtl*k3,uk,rk,tk);
    xkp1 = xk + (dtl/6)*(k1+2*k2+2*k3+k4);
end

function xkp1 = smoothint(xk,uk,rk,tk,dtl)
    %http://ancs.eng.buffalo.edu/pdf/ancs_papers/2013/geom_int.pdf
    a = [0          0           0           0               0           ;...
         0.8177227988124852 0   0           0               0           ;...
         0.3199876375476427 0.0659864263556022 0 0 0;
         0.9214417194464946  0.4997857776773573 -1.0969984448371582 0 0;...
         0.3552358559023322 0.2390958372307326  1.3918565724203246 -1.1092979392113565 0;];
     b = [0.1370831520630755;
         -0.0183698531564020;
         0.7397813985370780;
         -0.1907142565505889;
         0.3322195591068374];
     c = [0 0.8177227988124852 0.3859740639032449 0.3242290522866937  0.8768903263420429];

    dx0 = dynamics(xk,uk,rk,tk);
    dw0 = dx0(1:3);
    q0 = xk(4:7);
    w0 = xk(4:7);
    
    w1 = w0;
    q1 = q0;
    k1 = dw0;
    K1 = OmegaMat(w1);
    
    w2 = w0 + a(2,1)*(dtl)*k1;
    q2 = (eye(4,4)*cos(0.5*dtl*a(2,1)*norm(w1)) + (K1/norm(w1))*sin(0.5*dtl*a(2,1)*norm(w1)))*q1;
    q2 = q2/norm(q2);
    x2 = [w2;q2];
    dx2 = dynamics(x2,uk,rk,tk+c(2)*(dtl));
    k2 = dx2(1:3);
    K2 = OmegaMat(w2);
   
    w3 = w0 + a(3,2)*(dtl)*k2 + a(3,1)*(dtl)*k1;
    q3 = (eye(4,4)*cos(0.5*dtl*a(3,2)*norm(w2)) + (K2/norm(w2))*sin(0.5*dtl*a(3,2)*norm(w2)))*(eye(4,4)*cos(0.5*dtl*a(3,1)*norm(w1)) + (K1/norm(w1))*sin(0.5*dtl*a(3,1)*norm(w1)))*q1;
    q3 = q3/norm(q3);
    x3 = [w3;q3];
    dx3 = dynamics(x3,uk,rk,tk+c(3)*(dtl));
    k3 = dx3(1:3);
    K3 = OmegaMat(w3);
    
    
    w4 = w0 + a(4,:).'*dtl*[k1 k2 k3 zeros(3,2)];
    angs4 = dtl*0.5*a(4,:).*vecnorm([w1 w2 w3 w4 zeros(3,1)]);
    q4 = (eye(4,4)*cos(angs4(3)) + K3*sin(angs4(3))/norm(w3))*(eye(4,4)*cos(angs4(2)) + K2*sin(angs4(2))/norm(w2))*(eye(4,4)*cos(angs4(1)) + K1*sin(angs4(1))/norm(w1))*q1;
    q4 = q4/norm(q4);
    x4 = [w4;q4];
    dx4 = dynamics(x4,uk,rk,tk+c(4)*(dtl));
    k4 = dx4(1:3);
    K4 = OmegaMat(w4);
    
    
    
    w5 = w0 + a(5,:).'*dtl*[k1 k2 k3 k4 zeros(3,1)];
    angs5 = dtl*0.5*a(5,:).*vecnorm([w1 w2 w3 w4 w5]);
    q5 = (eye(4,4)*cos(angs5(4)) + K4*sin(angs5(4))/norm(w4))*(eye(4,4)*cos(angs5(3)) + K3*sin(angs5(3))/norm(w3))*(eye(4,4)*cos(angs5(2)) + K2*sin(angs5(2))/norm(w2))*(eye(4,4)*cos(ang5(1)) + K1*sin(angs5(1))/norm(w1))*q1;
    q5 = q5/norm(q5);
    x5 = [w5;q5];
    dx5 = dynamics(x5,uk,rk,tk+c(5)*(dtl));
    k5 = dx5(1:3);
    K5 = OmegaMat(w5);
    
    wkp1 = w0 + b.'*dtl*[k1 k2 k3 k4 k5];
    angskp1 = dtl*0.5*b.*vecnorm([w1 w2 w3 w4 w5]);
    qkp1 = (eye(4,4)*cos(angskp1(5)) + K5*sin(angskp1(5))/norm(w5))*(eye(4,4)*cos(angskp1(4)) + K4*sin(angskp1(4))/norm(w4))*(eye(4,4)*cos(angskp1(3)) + K3*sin(angskp1(3))/norm(w3))*(eye(4,4)*cos(angskp1(2)) + K2*sin(angskp1(2))/norm(w2))*(eye(4,4)*cos(angskp1(1)) + K1*sin(angskp1(1))/norm(w1))*q1;
    xkp1 = [wkp1;qkp1];
end

% function [Ak, Bk] = smoothintJac(xk,uk,rk,tk,dtl)
%     %http://ancs.eng.buffalo.edu/pdf/ancs_papers/2013/geom_int.pdf
%     a = [0          0           0           0               0           ;...
%          0.8177227988124852 0   0           0               0           ;...
%          0.3199876375476427 0.0659864263556022 0 0 0;
%          0.9214417194464946  0.4997857776773573 -1.0969984448371582 0 0;...
%          0.3552358559023322 0.2390958372307326  1.3918565724203246 -1.1092979392113565 0;];
%      b = [0.1370831520630755;
%          -0.0183698531564020;
%          0.7397813985370780;
%          -0.1907142565505889;
%          0.3322195591068374];
%      c = [0 0.8177227988124852 0.3859740639032449 0.3242290522866937  0.8768903263420429];
% 
%     
%     dx0 = dynamics(xk,uk,rk,tk);
%     dw0 = dx0(1:3);
%     q0 = xk(4:7);
%     w0 = xk(4:7);
%     
%     w1 = w0;
%     q1 = q0;
%     k1 = dw0;
%     K1 = OmegaMat(w1);
%     
%     w2 = w0 + a(2,1)*(dtl)*k1;
%     q2 = (eye(4,4)*cos(0.5*dtl*a(2,1)*norm(w1)) + (K1/norm(w1))*sin(0.5*dtl*a(2,1)*norm(w1)))*q1;
%     q2 = q2/norm(q2);
%     x2 = [w2;q2];
%     dx2 = dynamics(x2,uk,rk,tk+c(2)*(dtl));
%     k2 = dx2(1:3);
%     K2 = OmegaMat(w2);
%    
%     w3 = w0 + a(3,2)*(dtl)*k2 + a(3,1)*(dtl)*k1;
%     q3 = (eye(4,4)*cos(0.5*dtl*a(3,2)*norm(w2)) + (K2/norm(w2))*sin(0.5*dtl*a(3,2)*norm(w2)))*(eye(4,4)*cos(0.5*dtl*a(3,1)*norm(w1)) + (K1/norm(w1))*sin(0.5*dtl*a(3,1)*norm(w1)))*q1;
%     q3 = q3/norm(q3);
%     x3 = [w3;q3];
%     dx3 = dynamics(x3,uk,rk,tk+c(3)*(dtl));
%     k3 = dx3(1:3);
%     K3 = OmegaMat(w3);
%     
%     
%     w4 = w0 + a(4,:).'*dtl*[k1 k2 k3 zeros(3,2)];
%     angs4 = dtl*0.5*a(4,:).*vecnorm([w1 w2 w3 w4 zeros(3,1)]);
%     q4 = (eye(4,4)*cos(angs4(3)) + K3*sin(angs4(3))/norm(w3))*(eye(4,4)*cos(angs4(2)) + K2*sin(angs4(2))/norm(w2))*(eye(4,4)*cos(angs4(1)) + K1*sin(angs4(1))/norm(w1))*q1;
%     q4 = q4/norm(q4);
%     x4 = [w4;q4];
%     dx4 = dynamics(x4,uk,rk,tk+c(4)*(dtl));
%     k4 = dx4(1:3);
%     K4 = OmegaMat(w4);
%     
%     
%     
%     w5 = w0 + a(5,:).'*dtl*[k1 k2 k3 k4 zeros(3,1)];
%     angs5 = dtl*0.5*a(5,:).*vecnorm([w1 w2 w3 w4 w5]);
%     q5 = (eye(4,4)*cos(angs5(4)) + K4*sin(angs5(4))/norm(w4))*(eye(4,4)*cos(angs5(3)) + K3*sin(angs5(3))/norm(w3))*(eye(4,4)*cos(angs5(2)) + K2*sin(angs5(2))/norm(w2))*(eye(4,4)*cos(ang5(1)) + K1*sin(angs5(1))/norm(w1))*q1;
%     q5 = q5/norm(q5);
%     x5 = [w5;q5];
%     dx5 = dynamics(x5,uk,rk,tk+c(5)*(dtl));
%     k5 = dx5(1:3);
%     K5 = OmegaMat(w5);
%     
%     wkp1 = w0 + b.'*dtl*[k1 k2 k3 k4 k5];
%     angskp1 = dtl*0.5*b.*vecnorm([w1 w2 w3 w4 w5]);
%     qkp1 = (eye(4,4)*cos(angskp1(5)) + K5*sin(angskp1(5))/norm(w5))*(eye(4,4)*cos(angskp1(4)) + K4*sin(angskp1(4))/norm(w4))*(eye(4,4)*cos(angskp1(3)) + K3*sin(angskp1(3))/norm(w3))*(eye(4,4)*cos(angskp1(2)) + K2*sin(angskp1(2))/norm(w2))*(eye(4,4)*cos(angskp1(1)) + K1*sin(angskp1(1))/norm(w1))*q1;
%     xkp1 = [wkp1;qkp1];
%     dangsdu = dt1*0.5*b.*dot([w1 w2 w3 w4 w5],[dw1du dw2du dw3du dw4du dw5u])./vecnorm([w1 w2 w3 w4 w5]);
%     dangsdw = dt1*0.5*b.*dot([w1 w2 w3 w4 w5],[dw1dw dw2dw dw3dw dw4dw dw5w])./vecnorm([w1 w2 w3 w4 w5]);
%     dangsdq = dt1*0.5*b.*dot([w1 w2 w3 w4 w5],[dw1dq dw2dq dw3dq dw4dq dw5q])./vecnorm([w1 w2 w3 w4 w5]);
% %     dqdq
% %     dqdw
%     qmat = nan(4,4,5);
%     qmat(:,:,1) = (eye(4,4)*cos(angskp1(1)) + K1*sin(angskp1(1))/norm(w1));
%     qmat(:,:,2) = (eye(4,4)*cos(angskp1(2)) + K2*sin(angskp1(2))/norm(w2));
%     qmat(:,:,3) = (eye(4,4)*cos(angskp1(3)) + K3*sin(angskp1(3))/norm(w3));
%     qmat(:,:,4) = (eye(4,4)*cos(angskp1(4)) + K4*sin(angskp1(4))/norm(w4));
%     qmat(:,:,5) = (eye(4,4)*cos(angskp1(5)) + K5*sin(angskp1(5))/norm(w5));
%     dO1dwx = OmegaMat([1;0;0]);
%     dO2dwy = OmegaMat([0;1;0]);
%     dO3dwz = OmegaMat([0;0;1]);
%     dM1dang = eye(4,4)*(-sin(angskp1(1))) + K1*cos(angskp1(1))/norm(w1);
%     dM2dang = eye(4,4)*(-sin(angskp1(2))) + K2*cos(angskp1(2))/norm(w2);
%     dM3dang = eye(4,4)*(-sin(angskp1(3))) + K3*cos(angskp1(3))/norm(w3);
%     dM4dang = eye(4,4)*(-sin(angskp1(4))) + K4*cos(angskp1(4))/norm(w4);
%     dM5dang = eye(4,4)*(-sin(angskp1(5))) + K5*cos(angskp1(5))/norm(w5);
%     
%     dM1du = dM1dang*dangs(1
%     dM2du = eye(4,4)*(-sin(angskp1(2))) + K2*cos(angskp1(2))/norm(w2);
%     dM3du = eye(4,4)*(-sin(angskp1(3))) + K3*cos(angskp1(3))/norm(w3);
%     dM4du = eye(4,4)*(-sin(angskp1(4))) + K4*cos(angskp1(4))/norm(w4);
%     dM5dang = eye(4,4)*(-sin(angskp1(5))) + K5*cos(angskp1(5))/norm(w5);
%     
%     dqdu = dM5dang
%     
%     dqdu = (eye(4,4)*cos(angskp1(5)) + K5*sin(angskp1(5))/norm(w5))*(eye(4,4)*cos(angskp1(4)) + K4*sin(angskp1(4))/norm(w4))*(eye(4,4)*cos(angskp1(3)) + K3*sin(angskp1(3))/norm(w3))*(eye(4,4)*cos(angskp1(2)) + K2*sin(angskp1(2))/norm(w2))*(eye(4,4)*cos(angskp1(1)) + K1*sin(angskp1(1))/norm(w1))*q1;
%     
%     dwdq = dw0dq + b.'*dtl*[dk1dq dk2dq dk3dq dk4dq dk5dq];
%     dwdw = dw0dw + b.'*dtl*[dk1dw dk2dw dk3dw dk4dw dk5dw];
%     dwdu = dw0du + b.'*dtl*[dk1du dk2du dk3du dk4du dk5du];
%     
%     Ak = [dwdw dwdq; dqdw dqdq];
%     Bk = [dwdu;dqdu];
% end

function O = OmegaMat(w)
    arguments
        w (3,1)
    end
    O = 0.5*[0 w.'; w -skewsym(w)];
end

function Imu = Imu(mu,c,lam,useSlacks) %ineq before eq
    global numc numIneq cmax numcp
    if useSlacks
        ii = mu*ones(1,numcp);
        iszer = and(and((c <= 0),lam <= 0),(1:numcp).'<= numIneq);
    %     iszer = and(and((c <= -cmax),lam <= 0),(1:numc).'<= numIneq);
        ii(iszer) = 0;
        Imu = diag(ii);
    else
        ii = mu*ones(1,numc);
        iszer = and(and((c <= 0),lam <= 0),(1:numc).'<= numIneq);
    %     iszer = and(and((c <= -cmax),lam <= 0),(1:numc).'<= numIneq);
        ii(iszer) = 0;
        Imu = diag(ii);
    end
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

function [B_gram,B_cond] = magnetic_grammian(Bset)
    global N dt sim_length
    %this function takes in a magnetic field of size
    % Nx3 and returns the magnetic field gramian B^T*B
    %of size 3x3xN
    B_gram = zeros(3,3,N);
    B_cond = zeros(1,N);
    B1 = Bset(1, 1);
    B2 = Bset(2, 1);
    B3 = Bset(3, 1);
    B_gram_1 = [0 -B3 B2;
             B3  0 -B1;
              -B2 B1 0];
    B_gram(:, :, 1) = B_gram_1*B_gram_1;
    B_cond(1) = cond(B_gram(:, :, 1));
    for i=2:N
        B1 = Bset(1, dt*(i-1)+1);
        B2 = Bset(2, dt*(i-1)+1);
        B3 = Bset(3, dt*(i-1)+1);
        B_skew_i = [0 -B3 B2;
                   B3  0 -B1;
                   -B2 B1 0];
        B_gram_i = B_skew_i*B_skew_i*1;%*dt;
        B_gram(:, :, i) = B_gram(:, :, i-1) + B_gram_i;
        B_cond(i) = cond(B_gram(:, :, i));
    end
end

% function tf_index = condition_cutoff_time(B_gram,cutoff)
%     %this function takes in a 3x3xN array of magnetic field
%     %grammian and spits out the time index at which the cutoff
%     % is hit for the condition number
%     global N dt
%     B_gram_cond = zeros(1,N);
%     %generate 2-norm condition number of the grammian
%     for i = 1:N
%         B_gram_i = B_gram(:, :, i);
%         B_gram_cond(i) = cond(B_gram_i, 2);
%     end
%     %find first location where the condition of the grammian is below the
%     %cutoff
%     tf_index = round(1/eps);
%     for i = 1:N
%         if B_gram_cond(i) < cutoff && tf_index == 0
%             tf_index = i;
%         end
%     end
% end


function [Pset,Kset,dset,delV,rho,drho] = sqrtbackwardpass(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,rho,drho,mu,slackset,useSlacks)
    global N m n QN regScale regMin dt numDOF numc
    if useSlacks
        Kset = nan(m+n,numDOF,N);
        dset = nan(m+n,N-1);
    else
        Kset = nan(m,numDOF,N);
        dset = nan(m,N-1);
    end
    Pset = nan(numDOF,numDOF,N);
%     sPset = nan(numDOF,numDOF,N);
    delV = [0 0];
    
    k=N;
    [lkxx,lkuu,lkux,lkx,lku,ckx,cku,lkss,lks,cks] = costJacobians(k,Xset(:,k),zeros(m,1),Xdesset(:,k),Rset(:,dt*(k-1)+1),Vset(:,dt*(k-1)+1),zeros(n,1),useSlacks);
    ck = constraints(k,Xset(:,k),zeros(m,1),zeros(n,1),useSlacks);
    Imuk = Imu(mu,ck,lambdaSet(:,k),useSlacks);
    
    try 
        slkxx = chol(lkxx);
    catch
        if sum(diag(lkxx)) ~= 0
            [v,d] = eig(lkxx);
            slkxx = sqrt(d)*v.';
        else
            error("can't do sqrt :(")
        end
    end
        
    
    pk = lkx + ckx.'*(lambdaSet(:,k) + Imuk*ck);
    sPk = triu(qr([slkxx; sqrt(Imuk)*ckx],0));
    sPk = sPk(1:size(sPk,2),:);
  
    qk = Xset(end-3:end,end);
    Gk = Gmat(qk);
    
    pn = pk;
    sPn = sPk;
    Gn = Gk;
%     sPset(:,:,k) = sPk;
    Pk = sPk.'*sPk;
    Pn = Pk;
    Pset(:,:,k) = Pk;
    
    k = N-1;
    while k>0
        Gkp1 = Gk;
        Pkp1 = Pk;
        sPkp1 = sPk;
        pkp1 = pk;
        
        xk = Xset(:,k);
        xdesk = Xdesset(:,k);
        uk = Uset(:,k);
        qk = xk(end-3:end);
        if useSlacks
            sk = slackset(:,k);
        else
            sk = [];
        end
        tk = dt*(k-1)+1;
        rk = Rset(:,tk);
        vk = Vset(:,tk);
        
        Gk = Gmat(qk);
        [Ak,Bk] = rk4Jacobians(xk,uk,rk,tk);
        Aqk = Gkp1*Ak*Gk.';
        Bqk = Gkp1*Bk;
        if useSlacks
            Ck = eye(n,n);
            Cqk = Gkp1*Ck;
        else
            Cqk = [];
        end
        
        [lkxx,lkuu,lkux,lkx,lku,ckx,cku,lkss,lks,cks] = costJacobians(k,xk,uk,xdesk,rk,vk,sk,useSlacks);
        ck = constraints(k,xk,uk,sk,useSlacks);
        Imuk = Imu(mu,ck,lambdaSet(:,k),useSlacks);
        if useSlacks
            try 
                slkuu = chol([lkuu zeros(m,n); zeros(n,m) lkss]);
            catch
                if sum(diag([lkuu zeros(m,n); zeros(n,m) lkss])) ~= 0
                    [v,d] = eig([lkuu zeros(m,n); zeros(n,m) lkss]);
                    slkuu = sqrt(d)*v.';
                else
                    error("can't do sqrt :(")
                end
            end
            Zkuu = triu(qr([slkuu;sPkp1*[Bqk Cqk];sqrt(Imuk)*[cku cks]],0));
            Zkuu = Zkuu(1:size(Zkuu,2),:);
            Zkuureg = triu(qr([Zkuu;sqrt(rho)*eye(m+n,m+n)],0));
            Zkuureg = Zkuureg(1:size(Zkuureg,2),:);
        else
            try 
                slkuu = chol(lkuu);
            catch
                if sum(diag(lkuu)) ~= 0
                    [v,d] = eig(lkuu);
                    slkuu = sqrt(d)*v.';
                else
                    error("can't do sqrt :(")
                end
            end
            Zkuu = triu(qr([slkuu;sPkp1*Bqk;sqrt(Imuk)*cku],0));
            Zkuu = Zkuu(1:size(Zkuu,2),:);
            Zkuureg = triu(qr([Zkuu;sqrt(rho)*eye(m,m)],0));
            Zkuureg = Zkuureg(1:size(Zkuureg,2),:);
        end

        if cond(Zkuureg) > 50 %~all(real(eig(triu(Qkuureg) + triu(Qkuureg)'))) > 0
            %problem! Zkuu is not invertible! increase regularization and
            %re-try
            k = N-1;
            drho = max(drho*regScale,regScale);
            rho = max(rho*drho,regMin);
            
            pk = pn;
            Pk = Pn;
            sPk = sPn;
            Gk = Gn;
        else
            try 
                slkxx = chol(lkxx);
            catch
                if sum(diag(lkxx)) ~= 0
                    [v,d] = eig(lkxx);
                    slkxx = sqrt(d)*v.';
                else
                    error("can't do sqrt :(")
                end
            end
            Zkxx = triu(qr([slkxx;sPkp1*Aqk;sqrt(Imuk)*ckx],0));
            Zkxx = Zkxx(1:size(Zkxx,2),:);
            
                
            
            Qkux = [lkux;zeros(n*useSlacks,numDOF)] + [Bqk.'; Cqk.']*(sPkp1.'*sPkp1)*Aqk + [cku.'; cks.']*Imuk*ckx;
            Qkx = lkx + Aqk.'*pkp1 + ckx.'*(lambdaSet(:,k) + Imuk*ck);
            Qku = [lku;lks] + [Bqk.'; Cqk.']*pkp1 + [cku.'; cks.']*(lambdaSet(:,k) + Imuk*ck);
            
%             Kk = -Qkuureg\Qkux;
            Kk = -Zkuureg\(Zkuureg.'\Qkux);
            Kset(:,:,k) = Kk;
%             dk = -Qkuureg\Qku;
            dk = -Zkuureg\(Zkuureg.'\Qku);
            dset(:,k) = dk;
            pk = Qkx + (Kk.'*Zkuu.')*(Zkuu*dk) + Kk.'*Qku + Qkux.'*dk;

            delV = delV + [dk.'*Qku 0.5*(dk.'*Zkuu.')*(Zkuu*dk)];

            if cond(Zkxx) > 50
                t1 = Zkxx.'\Qkux.';
            else
                t1 = pinv(Zkxx.')*Qkux.';
            end
            try 
                t2 = chol(Zkuu.'*Zkuu - t1.'*t1);
            catch
%                 if sum(diag(lkxx)) ~= 0
                    [v,d] = eig(Zkuu.'*Zkuu - t1.'*t1);
                    t2 = sqrt(d)*v.';
%                 else
%                     error("can't do sqrt :(")
%                 end
            end
            sPk = [Zkxx + t1*Kk;t2*Kk];
            
%             Pk = Qkxx + Kk.'*Qkuu*Kk + Kk.'*Qkux + Qkux.'*Kk;
%             Pk = 0.5*(Pk+Pk.');
            Pk = sPk.'*sPk;
            Pset(:,:,k) = Pk;
            k = k-1;
        end
        
    end
    drho = min(drho/regScale,1/regScale);
    rho = rho*drho*(rho*drho>regMin);
end


function [Pset,Kset,dset,delV,rho,drho, Qkx_debug_matlab, Qku_debug_matlab] = backwardpass(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,rho,drho,mu,slackset,useSlacks)
    global N m n QN regScale regMin dt numDOF numc
    if useSlacks
        Kset = zeros(m+n,numDOF,N-1);
        dset = zeros(m+n,N-1);
    else
        Kset = zeros(m,numDOF,N-1);
        dset = zeros(m,N-1);
    end
    Pset = zeros(numDOF,numDOF,N);
    delV = [0 0];
    
    k=N;
    [lkxx,lkuu,lkux,lkx,lku,ckx,cku,lkss,lks,cks] = costJacobians(k,Xset(:,k),zeros(m,1),Xdesset(:,k),Rset(:,dt*(k-1)+1),Vset(:,dt*(k-1)+1),zeros(n,1),useSlacks);
    ck = constraints(k,Xset(:,k),zeros(m,1),zeros(n,1),useSlacks);
    Imuk = Imu(mu,ck,lambdaSet(:,k),useSlacks);
    
    pk = lkx + ckx.'*(lambdaSet(:,k) + Imuk*ck);
    Pk = lkxx + ckx.'*Imuk*ckx;
  
    qk = Xset(end-3:end,end);
    Gk = Gmat(qk);
    
    pn = pk;
    Pn = Pk;
    Gn = Gk;
    Pset(:,:,k) = Pk;
    
    k = N-1;
    Qkx_debug_matlab = zeros(6, N);
    Qku_debug_matlab = zeros(3, N);
    while k>0
        Gkp1 = Gk;
        Pkp1 = Pk;
        pkp1 = pk;
        
        xk = Xset(:,k);
        xdesk = Xdesset(:,k);
        uk = Uset(:,k);
        qk = xk(end-3:end);
        if useSlacks
            sk = slackset(:,k);
        else
            sk = [];
        end
        tk = dt*(k-1)+1;
        rk = Rset(:,tk);
        vk = Vset(:,tk);
        
        Gk = Gmat(qk);
        [Ak,Bk] = rk4Jacobians(xk,uk,rk,tk);
        Aqk = Gkp1*Ak*Gk.';
        Bqk = Gkp1*Bk;
        if useSlacks
            Ck = eye(n,n);
            Cqk = Gkp1*Ck;
        else
            Cqk = [];
        end
        
        [lkxx,lkuu,lkux,lkx,lku,ckx,cku,lkss,lks,cks] = costJacobians(k,xk,uk,xdesk,rk,vk,sk,useSlacks);
        ck = constraints(k,xk,uk,sk,useSlacks);
        Imuk = Imu(mu,ck,lambdaSet(:,k),useSlacks);
        if useSlacks
            Qkuu = [lkuu zeros(m,n); zeros(n,m) lkss] + [Bqk.'; Cqk.']*Pkp1*[Bqk Cqk] + [cku.'; cks.']*Imuk*[cku cks];
            Qkuureg = Qkuu + rho*eye(m+n,m+n);
        else
            Qkuu = lkuu + Bqk.'*Pkp1*Bqk + cku.'*Imuk*cku;
            Qkuureg = Qkuu + rho*eye(m,m);
        end
%         if cond(Qkuureg) > 50 
        if ~all(real(eig(triu(Qkuureg) + triu(Qkuureg)'))) > 0
            %problem! Qkuu is not invertible! increase regularization and
            %re-try
            k = N-1;
            drho = max(drho*regScale,regScale);
            rho = max(rho*drho,regMin);
            
            pk = pn;
            Pk = Pn;
            Gk = Gn;
            maxQ=max(abs(Qkuu),[],'all')
            maxQreg = max(abs(Qkuureg),[],'all')
            avgQ=mean(abs(Qkuu),'all')
            avgQreg = mean(abs(Qkuureg),'all')
        else
            Qkxx = lkxx + Aqk.'*Pkp1*Aqk + ckx.'*Imuk*ckx;
            Qkux = [lkux;zeros(n*useSlacks,numDOF)] + [Bqk.'; Cqk.']*Pkp1*Aqk + [cku.'; cks.']*Imuk*ckx;
            Qkx = lkx + Aqk.'*pkp1 + ckx.'*(lambdaSet(:,k) + Imuk*ck);
            Qkx_debug_matlab(:, k) = Qkx;
            Qku = [lku;lks] + [Bqk.'; Cqk.']*pkp1 + [cku.'; cks.']*(lambdaSet(:,k) + Imuk*ck);
            Qku_debug_matlab(:, k) = Qku;
            Qkuureg_inv = inv(Qkuureg);
            Kk = -Qkuureg_inv*Qkux;
            Kset(:,:,k) = Kk;
            dk = -Qkuureg_inv*Qku;
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

function [newXset,newUset,newLA,rho,drho,newslackset] = forwardpass(Xset,Uset,Kset,Rset,Vset,dset,delV,LA,lambdaSet,rho,drho,mu,Xdesset,slackset,useSlacks)
    global N maxLsIter beta1 beta2 regScale regMin regBump maxState maxControl
    alph = 1;
    newLA = 1/eps^2;
    z = -1;
    
    iter = 0;
    while ((z<=beta1 || z>beta2) && (newLA>=LA))
        if iter > maxLsIter
            'no update in forward pass'
            newXset = Xset;
            newUset = Uset;
            newslackset = slackset;
            newLA = cost2(newXset,newUset,Rset,Vset,Xdesset,lambdaSet,mu,slackset,useSlacks);
            z = 0;
            alph = 0;
            exp = 0;
            rho = rho + regBump;
            drho = max(drho*regScale,regScale);
            rho = max(rho*drho,regMin);
            break
        end
        
        [newXset,newUset,newslackset] = generateTrajectory(Xset,Uset,Kset,dset,Rset,Vset,alph,lambdaSet,slackset,useSlacks);
        
        nonfinflag = any([any(isnan(newXset),'all'),any(isnan(newUset),'all'),any(isnan(newslackset),'all'),any(isinf(newXset),'all'),any(isinf(newUset),'all'),any(isinf(newslackset),'all'),any(abs(newXset) > maxState,'all'),any(abs(newUset) > maxControl,'all')]);
        if nonfinflag
            %'nan or inf in forward pass'
%             max(Kset,[],'all')
%             max(dset,[],'all')
%             nnz(isnan(newXset))
%             nnz(isnan(newUset))
            iter = iter + 1;
            alph  = alph/2;
            iter;
            alph;
            continue
        end
        iter;
        alph;
        newLA = cost2(newXset,newUset,Rset,Vset,Xdesset,lambdaSet,mu,newslackset,useSlacks);
        if newLA <0
            error('negative cost somehow')
        end
        
        exp = -alph*(delV(1) + alph*delV(2));
        if exp > 0
            z = (LA-newLA)/exp;
        else
            z = -1;
        end
        iter = iter + 1;
        alph  = alph/2;
%         z
%         newLA-LA
    end
    
    if newLA > LA
        newXset = newXset*nan(1,1);
        error('cost increased in forward pass')
    end
end

function [newX,newU,newslack] = generateTrajectory(Xset,Uset,Kset,dset,Rset,Vset,alph,lambdaSet,slackset,useSlacks)
    global n m N dt randnoise
    newX = zeros(n,N);
    newU = zeros(m,N-1);
    if useSlacks
        newslack = nan(n,N-1);
    end
    
    if useSlacks
        newX(:,1) = Xset(:,1);
        for k = 2:N
            %delrem = newX(1:end-4,k-1) - Xset(1:end-4,k-1);
            %delq = [dot(newX(end-3:end,k-1),Xset(end-3:end,k-1)); Xset(end-3,k-1)*newX(end-2:end,k-1)-newX(end-3,k-1)*Xset(end-2:end,k-1) - cross(Xset(end-2:end,k-1),newX(end-2:end,k-1))];
            %delang = delq(2:end)/(1+delq(1));
%             newU(:,k-1) = Uset(:,k-1) + Kset(:,:,k-1)*[delrem;delang] + alph*dset(:,k-1);
            delx = Gmat(Xset(end-3:end,k-1))*(newX(:,k-1) - Xset(:,k-1));%*(newX(:,k-1) - [Xset(1:end-4,k-1);zeros(4,1)]); %I am not confident in this. Look at equation 81 in the AL_iLQR Tutorial by Jackson

%             delx = Gmat(Xset(end-3:end,k-1))*xdiff(newX(:,k-1), Xset(:,k-1),k,Rset(:,k),Vset(:,k));%*(newX(:,k-1) - [Xset(1:end-4,k-1);zeros(4,1)]); %I am not confident in this. Look at equation 81 in the AL_iLQR Tutorial by Jackson
            delus = Kset(:,:,k-1)*delx + alph*dset(:,k-1);

            newU(:,k-1) = Uset(:,k-1) + delus(1:m);
            newU(:,k-1) = newU(:,k-1)*(1-randnoise+2*randnoise*rand(1,1));
            newslack(:,k-1) = slackset(:,k-1) + delus(m+1:end);
            xk = rk4(newX(:,k-1),newU(:,k-1),Rset(:,dt*(k-2)+1),dt*(k-2)+1,dt) + newslack(:,k-1);
            xk(end-3:end) = xk(end-3:end)/norm(xk(end-3:end));
            newX(:,k) = xk;
        end
    else
        newX(:,1) = Xset(:,1);
        for k = 2:N
            %delrem = newX(1:end-4,k-1) - Xset(1:end-4,k-1);
            %delq = [dot(newX(end-3:end,k-1),Xset(end-3:end,k-1)); Xset(end-3,k-1)*newX(end-2:end,k-1)-newX(end-3,k-1)*Xset(end-2:end,k-1) - cross(Xset(end-2:end,k-1),newX(end-2:end,k-1))];
            %delang = delq(2:end)/(1+delq(1));
            %newU(:,k-1) = Uset(:,k-1) + Kset(:,:,k-1)*[delrem;delang] + alph*dset(:,k-1);
            delx = Gmat(Xset(end-3:end,k-1))*(newX(:,k-1) - Xset(:,k-1));%*(newX(:,k-1) - [Xset(1:end-4,k-1);zeros(4,1)]); %I am not confident in this. Look at equation 81 in the AL_iLQR Tutorial by Jackson
%             delx = Gmat(Xset(end-3:end,k-1))*xdiff(newX(:,k-1), Xset(:,k-1),k,Rset(:,k),Vset(:,k));%*(newX(:,k-1) - [Xset(1:end-4,k-1);zeros(4,1)]); %I am not confident in this. Look at equation 81 in the AL_iLQR Tutorial by Jackson
            delu = Kset(:,:,k-1)*delx + alph*dset(:,k-1);
            newU(:,k-1) = Uset(:,k-1) + delu;
            Uprev = Uset(:,k-1);
            xk = rk4(newX(:,k-1),newU(:,k-1),Rset(:,dt*(k-2)+1),dt*(k-2)+1,dt);
            xk(end-3:end) = xk(end-3:end)/norm(xk(end-3:end));
            newX(:,k) = xk;
        end
        newslack = [];
    end
end

function dx = xdiff(x1,x0,k,rk,vk)
    dx = 0*x1;
    dx(1:end-4) = x1(1:end-4)-x0(1:end-4);
    [v,u] = pvs(k,rk,vk);
    q1 = x1(end-3:end);
    [~,~,q0] = cost2ang(q1,v,u);
    dx(end-3:end) = q1-q0;
end


function [newX,newslack] = generateTrajectory0(x0,Uset,Rset,Vset,trajGuess,useSlacks,dtl,Nl)
    global n useGuess
    newX = zeros(n,Nl);
    
    if useSlacks
        newslack = zeros(n,Nl-1);
        newX = trajGuess;
        for k = 1:Nl-1
            %newX(:,k) = rk4(newX(:,k-1),Uset(:,k-1),Rset(:,k-1),k*dt); 
            xkp10 = rk4(trajGuess(:,k),Uset(:,k),Rset(:,dtl*(k-1)+1),dtl*(k-1)+1,dtl); 
            newslack(:,k) = newX(:,k+1) - xkp10;
    %         if any(isnan(newX(:,k)))
    %             rk4(newX(:,k-1),Uset(:,k-1),Rset(:,dt*(k-1)),(k-1)*dt)
    %         end
        end
    else
        newX(:,1) = x0;
        for k = 2:Nl
            %newX(:,k) = rk4(newX(:,k-1),Uset(:,k-1),Rset(:,k-1),k*dt); 
            xk = rk4(newX(:,k-1),Uset(:,k-1),Rset(:,dtl*(k-2)+1),dtl*(k-2)+1,dtl); 
            xk(end-3:end) = xk(end-3:end)/norm(xk(end-3:end));
            newX(:,k) = xk;
    %         if any(isnan(newX(:,k)))
    %             rk4(newX(:,k-1),Uset(:,k-1),Rset(:,dt*(k-1)),(k-1)*dt)
    %         end
        end
        newslack = [];
    end
end

function newy = projection(H0,Xset,Uset,lambdaSet,mu)
    global m N n numc projMaxRef projConvRateThreshold projRegPrimal projRegChol projcmax projMaxIter cmax
    Uset = [Uset zeros(m,1)];
    y = [Xset;Uset];
    y = y(:);
    [v0,cv0,active,clist] = maxViol(Xset,Uset,lambdaSet,mu);
    v0 = max(v0,cv0);
    H = diag(diag(H0)); %H0 is hessian of just the cost?, making it diagonal here

    clist = sparse(clist);
    active = sparse(active);
    y = sparse(y);

    if projRegPrimal > 0
        dimPrim = size(y,1);
        zz = zeros(1,length(H));
        zz(1:dimPrim) = projRegPrimal;
        H = H + diag(zz);
    end
    
    for i = 1:projMaxIter
        D = spalloc(N*numc,N*(m+n)-m,nnz(active)*(m+n));
        d = spalloc(N*numc,nnz(active));
        dloc = zeros(nnz(active),1);
        Dloc = zeros(nnz(active)*(m+n),2);
        dval = zeros(nnz(active),1);
        Dval = zeros(nnz(active)*(m+n),1);
        k = N;
        dloc = nonzeros(active(:));
        
        d((N-1)*numc+1:(N-1)*numc+numc) = -clist(:,k);
        [~,ckx] = constraintJacobians(N,Xset(:,N),zeros(m,1));
        D((N-1)*numc+1:(N-1)*numc+numc,(m+n)*(N-1)+1:(m+n)*(N-1)+n) = ckx;
        for k = 1:N
            d((k-1)*numc+1:(k-1)*numc+numc) = -clist(:,k);
            [cku,ckx] = constraintJacobians(k,Xset(:,k),Uset(:,k));
            D((k-1)*numc+1:(k-1)*numc+numc,(m+n)*(k-1)+1:(m+n)*(k-1)+(m+n)) = [ckx cku];
        end
        d = d(active(:));
        D = D(active(:),:);
        
        HinvD = H\D.';
        tt = D*HinvD;
        S = triu(tt) +  triu(tt,1).';
        Sreg = chol(S + eye(length(S),length(S))*projRegChol);
        
        vprev = v0;
        for c = 1:projMaxRef
            [newy,v] = linesearch(d,y,HinvD,S,Sreg,active);%linesearch(d0,y,HinvD,S,Sreg);
            convRate = log10(v)/log10(vprev);
            vprev = v;

            y = newy;
            if (convRate < projConvRateThreshold || v < cmax )
                break
            end
        end
        
%         lambda  = lambdaSet(active);
%         re-find D,d?
%         g
%         res0 = g + D.'*lambda;
%         A = D*D.';
%         Areg = A + eye(length(A),length(A))*projRegPrimal;
%         b = D*res0;
%         dlam = -regsolve(A,b,Areg,1e-10,10);
%         lambda = lambda + dlam;
%         res = norm(g + D.'*lambda);
        
        if v < cmax%projcmax
            break
        end
        
    end
    
end

function [newy,v] = linesearch(d,y,HinvD,S1,S2,active)
    global projLSRegTol projLSRegIter projLSIter m N n numc
    alph = 1;
    yy = reshape(y,[n+m,N]);
    Xset = yy(1:n,:);
    Uset = yy(n+1:end,1:end-1);
    lambda
    mu
    [v0,cv0,~,~] = maxViol(Xset,Uset,zeros(numc,N),0);
    v0 = max(v0,cv0);
%     dvec = d0(:);
%     d  = dvec(active(:));
    v = Inf;
    for c = 1:projLSIter
       dlam = regsolve(S1,d,S2,projLSRegTol,projLSRegIter);
       dy = -HinvD*dlam;
       newy = y + dy*alph;
       x
       u
       lambda
       mu
       [~,~,~,clist] = maxViol(newXset,newUset,lambdaSet,mu);
        for k = 1:N
            d((k-1)*numc+1:(k-1)*numc+numc) = -clist(:,k);
        end
       d = d(active(:));
       v = max(abs(d));
       if v < v0
           break
       end
       alph = alph/2;
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

function [Xset,Uset,lambdaSet,Kset,Pset,slackX,slackU] = trajOpt(x0,Rset,Vset,Xdesset,trajGuess)
    global m N useGuess numc numcp penInit lagMultInit U0 usePresetU useSQRT
    global LA0 cmax0 slackcmax slackLA slackLAnc LAdone LAdonenc cmaxdone LAnc0 dt
    global figlabel useGraph satAlignVector1 umax
    
    Urand = umax.*rand(m,N-1)/10;
    if ~usePresetU
        Uset = Urand;
        U0 = Uset;
    else
        if isempty(U0)
            Uset = Urand;
            U0 = Uset;
        else
            if length(U0) ~= N-1
                error('Existing Preset U is of wrong dimensions');
            end
            Uset = U0;
        end
    end
   
    [Xset,slackset] = generateTrajectory0(x0,Uset,Rset,Vset,trajGuess,useGuess,dt,N);
    nonfinflag = any([any(isnan(Xset),'all'),any(isnan(Uset),'all'),any(isinf(Xset),'all'),any(isinf(Uset),'all')]);
    if nonfinflag
        Xset
        Uset
        error('invalid first control sequence')
    end
    
    if useGuess
        lSet = lagMultInit*ones(numcp,N);
    else
        lSet = lagMultInit*ones(numc,N);
    end
    
    LA0 = cost(Xset,Uset,Rset,Vset,Xdesset,lSet,penInit,slackset,useGuess)
    LAnc0 = cost(Xset,Uset,Rset,Vset,Xdesset,0*lSet,0*penInit,slackset,useGuess)
    [cmax0,slackmax0,~,~] = maxViol(Xset,Uset,slackset,lSet,1,useGuess);
    cmax0 = max(cmax0,slackmax0)
        
    [Xset,Uset,lambdaSet,Kset,Pset,mu,rho,drho,slackset] = alilqr(Xset,Uset,Rset,Vset,Xdesset,slackset,useGuess,false,[],[]);
    if useGuess
        slackLA = cost(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,mu,slackset,useGuess)
        slackLAnc = cost(Xset,Uset,Rset,Vset,Xdesset,0*lambdaSet,0*mu,slackset,useGuess)
        [slackcmax,slackslackmax,~,~] = maxViol(Xset,Uset,slackset,lambdaSet,mu,useGuess);
        slackcmax = max(slackcmax,slackslackmax)
        LA0
        cmax0
        
        slackX = Xset;
        slackU = Uset;
        
        if useGraph
            figlabel = figure;
            figure(figlabel);
            subplot(3,1,1)
            plot(quatrotate(quatconj(Xset(4:7,:).'),satAlignVector1.'));
            if useGuess
                subplot(3,1,2)
                plot(vecnorm(slackset).')
                subplot(3,1,3)
                plot(slackset.')
                title('live running, first run w/0 slacks') 
            else
                subplot(3,1,2)
                plot(Xset(1:3,:).')
                subplot(3,1,3)
                plot(Uset.')
            end
            drawnow
        end
        
        %fix solution without slacks
        if useSQRT
            [Pset,Kset,dset,~,~,~] = sqrtbackwardpass(Xset,Uset,Rset,Vset,Xdesset,lambdaSet(1:numc,:),rho,drho,mu,Xset(:,1:end-1)*0,false);
        else
            [Pset,Kset,dset,~,~,~] = backwardpass(Xset,Uset,Rset,Vset,Xdesset,lambdaSet(1:numc,:),rho,drho,mu,Xset(:,1:end-1)*0,false);
        end
        [Xset0,Uset0,~] = generateTrajectory(Xset,Uset,Kset,dset,Rset,Vset,0,lambdaSet,Xset(:,1:end-1)*0,false);
        
        if useGraph
            figlabel = figure;
        end
        %re-solve with new solution but no slacks. This is a warm start and
        %should hopefully be fast.
        [Xset,Uset,lambdaSet,Kset,Pset,mu,rho,drho,slackset] = alilqr(Xset0,Uset0,Rset,Vset,Xdesset,[],false,true,lambdaSet(1:numc,:),mu);
        
    else
        slackLAnc = nan;
        slackX = nan;
        slackU = nan;
        slackLA = nan;
        slackcmax = nan;
    end
    'starting values:'
    LA0
    LAnc0
    cmax0
    'slack values:'
    slackLA
    slackLAnc
    slackcmax
    'final values:'
    LAdone = cost(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,mu,0*slackset,false)
    LAdonenc = cost(Xset,Uset,Rset,Vset,Xdesset,0*lambdaSet,0*mu,0*slackset,false)
    [cmaxdone,slackmaxdone,~,~] = maxViol(Xset,Uset,[],lambdaSet,mu,false);
    cmaxdone = max(cmaxdone,slackmaxdone)
    %[Xset,Uset,lambdaSet,Kset] = projection(Xset,Uset,lambdaSet,Kset);
    
end

function [maxViol,maxSlack,activeSet,clist] = maxViol(Xset,Uset,slackset,lambdaSet,mu,useSlacks)
    global numc N cmax m n numcp slackmax
    if useSlacks
        clist = nan(numcp,N);
    else
        clist = zeros(numc,N);
        maxSlack = 0;
    end
    for k = 1:N-1
        lamk = lambdaSet(:,k);
        if useSlacks
            sk = slackset(:,k);
        else
            sk = [];
        end
        ck = constraints(k,Xset(:,k),Uset(:,k),sk,useSlacks);
        Imuk = Imu(mu,ck,lamk,useSlacks);
        clist(:,k) = (Imuk>0)*ck;
    end
    lamk = lambdaSet(:,N);
    ck = constraints(N,Xset(:,N),zeros(m,1),zeros(n,1),useSlacks);
    Imuk = Imu(mu,ck,lamk,useSlacks);
    clist(:,N) = (Imuk>0)*ck;
    maxViol = max(abs(clist),[],'all');
    if useSlacks
        maxSlack =  max(abs(clist(numc+1:end,:)),[],'all');
    end
    activeSet = [abs(clist(1:numc,:)) >= cmax; abs(clist(numc+1:end,:)) >= slackmax];
end

function [Xset,Uset,lambdaSet,Kset,Pset,mu,rho,drho,slackset] = alilqr(Xset,Uset,Rset,Vset,Xdesset,slackset,useSlacks,warmstart,lambdaIn,muIn)
    global lagMultInit numc N penInit regInit maxOuterIter maxIlqrIter gradTol costTol cmax zcountLim maxIter penMax penScale m numIneq lagMultMin lagMultMax ilqrCostTol slackcmax numcp useSQRT slackmax sv1 figlabel useGraph satAlignVector1 useExpAngle useAngSquared
%     global LA0
    iter = 0;
    %dLA = 1/eps;
    grad = 1/eps;
    %delV = [1/eps,1/eps];
    if warmstart
        lambdaSet = lambdaIn;
        mu = muIn;
        zcountLim0 = max(3,round(0.2*zcountLim));
    else
        if useSlacks
            lambdaSet = lagMultInit*ones(numcp,N);
        else
            lambdaSet = lagMultInit*ones(numc,N);
        end
        mu = penInit;
        zcountLim0 = zcountLim;
    end
    
    %AL
    for j = 1:maxOuterIter
        'j='
        j
        cmaxtmp = 0;
        dlaZcount = 0;
        LA = cost(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,mu,slackset,useSlacks);
        %LA0 = LA;
        iter = iter + 1;
        %ILQR
        rho = regInit;
        drho = regInit;
        for ii = 1:maxIlqrIter
            ii
            rho
            drho
            if useSQRT
                [Pset,Kset,dset,delV,rho,drho] = sqrtbackwardpass(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,rho,drho,mu,slackset,useSlacks);
            else
                [Pset,Kset,dset,delV,rho,drho] = backwardpass(Xset,Uset,Rset,Vset,Xdesset,lambdaSet,rho,drho,mu,slackset,useSlacks);
            end
            %if(j==4 && ii==1)
                %save('/Users/alexmeredith/starlab-beavercube-adcs/clean-rpi/beavercube-adcs/bc/matfiles/testBackwardPassNonzero.mat', 'Xset', 'Uset', 'lambdaSet', 'dset');
            %end
            Kset(:, :, 1)
            rho
            drho
            delV
            [newXset,newUset,newLA,rho,drho,newslackset] =  forwardpass(Xset,Uset,Kset,Rset,Vset,dset,delV,LA,lambdaSet,rho,drho,mu,Xdesset,slackset,useSlacks);

            grad = mean(max(abs(dset)./(abs([newUset;newslackset])+1),[],1));
            iter = iter + 1;
            dLA = abs(newLA-LA);
            rho
            drho
            dLA
            newLA
            if(j==5&&ii==13)
                Xset
                newXset
            end
            dlaZcount = (dLA == 0)*(dlaZcount + (dLA == 0));
            rho;
%             clist = nan(numc,N);
%             for k = 1:N-1
%                 lamk = lambdaSet(:,k);
%                 ck = constraints(k,newXset(:,k),newUset(:,k));
%                 Imuk = Imu(mu,ck,lamk);
%                 clist(:,k) = (Imuk>0)*ck;
%             end
%             lamk = lambdaSet(:,N);
%             ck = constraints(N,newXset(:,N),zeros(m,1));
%             Imuk = Imu(mu,ck,lamk);
%             clist(:,N) = (Imuk>0)*ck;
%             cmaxtmp = max(max(abs(clist)))
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
%             rho
%             drho
            
            Xset = newXset;
            Uset = newUset;
            slackset = newslackset;
            LA = newLA;
            LAnc = cost(Xset,Uset,Rset,Vset,Xdesset,0*lambdaSet,0,slackset*0,useSlacks);
            if useExpAngle
                approxAng = log(sqrt(LAnc/N/sv1));
            elseif useAngSquared
                approxAng = sqrt(sqrt(LAnc/N/sv1));
            else
                approxAng = sqrt(LAnc/N/sv1);
            end
            useGraph = 0;
            slackmax = 1;
            if useGraph
                figure(figlabel);
                subplot(3,1,1)
                plot(quatrotate(quatconj(Xset(4:7,:).'),satAlignVector1.'));
                title(['live running: ', num2str(j), ', ' num2str(ii) ])
                if useSlacks
                    subplot(3,1,2)
                    plot(vecnorm(slackset).')
                    subplot(3,1,3)
                    plot(slackset.')
                else
                    subplot(3,1,2)
                    plot(Xset(1:3,:).')
                    subplot(3,1,3)
                    plot(Uset.')
                end
                drawnow
            end

            if LA<0
                error('LA negative somehow')
            end
            [cmaxtmp,slackmaxtmp,~,clist] = maxViol(Xset,Uset,slackset,lambdaSet,mu,useSlacks);
            zcountLim0;
            if useSlacks
                slackmaxtmp
            end
            '------------------';
%             if ((((cmaxtmp<cmax && slackmaxtmp<slackmax) || j < maxOuterIter) && grad<gradTol && 0 < dLA && dLA < ilqrCostTol) ||...
%                     dlaZcount >= zcountLim0 )
            if ((((cmaxtmp<cmax&& slackmaxtmp<slackmax) || j < maxOuterIter) && grad<gradTol) ||...
                    dlaZcount > zcountLim0 ||...
                    (0 < dLA && dLA < ilqrCostTol && ((cmaxtmp<cmax && slackmaxtmp<slackmax) || j < maxOuterIter ) ))
%             if (dlaZcount > zcountLim0 ||...
%                     (0 <= dLA && dLA < ilqrCostTol && grad < gradTol ))
                'exit inner loop'
                'j='
                j
                'ii='
                ii
                'cmaxtmp'
                cmaxtmp
                'grad'
                grad
                'dLA'
                dLA
                break
            end
            
        end
        % end of inner loop
            
        %LA = cost(Xset,Uset,Rset,Xdesset,lambdaSet,mu);
%         if (cmaxtmp<cmax || mu >= penMax)
        if ((cmaxtmp<cmax && slackmaxtmp<slackmax) && ( mu >= penMax || (0 < dLA && dLA < costTol && grad < gradTol)))
%         if (cmaxtmp<cmax && ((0 <= dLA && dLA < costTol) || grad < gradTol))
            'exit outer loop'
            break
        end
        
        if useSlacks
            for k = 1:N
                for i = 1:numcp
                    lambdaSet(i,k) = max(-lagMultMax,min(lagMultMax,lambdaSet(i,k) + mu*clist(i,k)));
                    if i <= numIneq
                        lambdaSet(i,k) = max(0,lambdaSet(i,k));
                    end
                end
            end
        else
            for k = 1:N
                for i = 1:numc
                    lambdaSet(i,k) = max(-lagMultMax,min(lagMultMax,lambdaSet(i,k) + mu*clist(i,k)));
                    if i <= numIneq
                        lambdaSet(i,k) = max(0,lambdaSet(i,k));
                    end
                end
            end
        end
        mu = max(0,min(penMax, penScale*mu));
    end
    
    if (j >= maxOuterIter) && (ii >= maxIlqrIter) && ~useSlacks 
        
        [cmaxtmp,~, activeset,~] = maxViol(Xset,Uset,[],lambdaSet,mu,false);
        
        warning('Solver exited maximum iterations reached. Constraints may not be within tolerances.')
        "number of constraints still active (across all timesteps"
        numactive = sum(activeset,'all')
        "total number of constraints (across all timesteps"
        constraintsNum = numel(activeset)
        "Maximum constraints violation"
        cmaxtmp
    end 
            
end

function [Kset,Sset] = TVLQRconst(Xset,Uset,Rset,Vset)
    global N m n QN  dt numDOF R
    Kset = zeros(m,numDOF,N);
    Sset = zeros(numDOF,numDOF,N);
    
    k=N;
    tk = dt*(k-1)+1;
    rk = Rset(:,tk);
    xk = Xset(:,k);
    uk = zeros(m,1);
    vk = Vset(:,tk);
    qk = xk(end-3:end);
    
    [lkxx,lkuu,lkux,lkx,lku,ckx,cku,~,~,~] = costJacobians(k,xk,uk,0*xk,rk,vk,[],false);

    Sk = lkxx;
    Sset(:,:,k) = Sk;
  
    Gk = Gmat(qk);
    
    for k = N-1:-1:1
        Gkp1 = Gk;
        tk = dt*(k-1)+1;
        rk = Rset(:,tk);
        xk = Xset(:,k);
        uk = Uset(:,k);
        vk = Vset(:,tk);
        qk = xk(end-3:end);
        Skp1 = Sk;
        
        [lkxx,lkuu,lkux,lkx,lku,ckx,cku,~,~,~] = costJacobians(k,xk,uk,0*xk,rk,vk,[],false);
        Gk = Gmat(qk);
        [Ak,Bk] = rk4Jacobians(xk,uk,rk,tk);
        Aqk = Gkp1*Ak*Gk.';
        Bqk = Gkp1*Bk;
        
        Kk = (R+Bqk.'*Skp1*Bqk)\(Bqk.'*Skp1*Aqk);
        Kset(:,:,k) = Kk;
        
%         Sk = lkxx + Kk.'*R*Kk + (Aqk-Bqk*Kk)\Skp1*(Aqk-Bqk*Kk);
        Sk = lkxx + Aqk.'*Skp1*Aqk - Aqk.'*Skp1*Bqk*((Bqk.'*Skp1*Bqk + R)\Bqk.'*Skp1*Aqk);
        Sk = 0.5*(Sk+Sk.');
        Sset(:,:,k) = Sk;
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
function wbias_random = generate_wbias(w_bias_mean, w_bias_var)
    wbias_random = zeros(3, 1);
    wbias_random(1) = normrnd(w_bias_mean, w_bias_var);
    wbias_random(2) = normrnd(w_bias_mean, w_bias_var);
    wbias_random(3) = normrnd(w_bias_mean, w_bias_var);
end

function dx = orbprop(x)
    mu = 3.986004418*10^5; %km^3s^-2
    v = x(1:3);
    r = x(4:6);
    dr = v;
    dv = -r*mu/(norm(r)^3); %should add J2 eventually
    dx = [dv;dr];
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