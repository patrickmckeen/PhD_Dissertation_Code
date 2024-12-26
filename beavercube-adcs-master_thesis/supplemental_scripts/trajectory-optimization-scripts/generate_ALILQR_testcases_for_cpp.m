%Imu(0.5, [-0.18; -0.18; -0.18; 0.18; 0.18; 0.18], -0.1)
qtest =[0.1826; 0.3651; 0.5477; 0.7303];
qtest2 = [0.1104; 0.4417; 0.7730; -0.4417];
x = [-0.2; 0.5; 0.01; qtest];
u = [0.002; 0.005; 0.001];
r = zeros(3, 1);
t = 1;
[jacX,jacU] = dynamicsJacobians(x,u,r,t)
lambdaSet = zeros(3, 1000);
%[Pset,Kset,dset,delV,rho,drho] = backwardpass(xtraj,Utestcpp,Rset,Xdesset,lambdaSet,rho,drho,mu)
function Imu = Imu(mu,c,lam) %ineq before eq
    global numc numIneq cmax
    ii = mu*ones(1,numc)
    and((c <= 0),lam <= 0)
    iszer = and(and((c <= 0),lam <= 0),(1:numc).'<= numIneq)
%     iszer = and(and((c <= -cmax),lam <= 0),(1:numc).'<= numIneq);
    ii(iszer) = 0;
    Imu = diag(ii);
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
function vx = skewsym(v)
    vx=[0 -v(3) v(2) ; v(3) 0 -v(1) ; -v(2) v(1) 0 ];
end

function W = Wmat(q)
    W = [-q(2:end).'; q(1)*eye(3,3) + skewsym(q(2:end))];
end

function xdot = dynamics(x,u,r,t) %continuous dynamics
    global J B_ECI
    w = x(1:3);
    q = x(end-3:end);
    B = B_ECI(t, :).';%magneticFieldECI(r,t);
    %dtau = distTorq(r,x);
    wdot = -J\(cross(w,J*w) + cross(rotT(q)*B,u)); %- dtau);
    qdot = 0.5*Wmat(q)*w;
    xdot = [wdot;qdot];
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
function B = magneticFieldECI(rECI,t)
    global B_ECI dt
    B = B_ECI(t,:).';
end
function [jacX,jacU] = dynamicsJacobians(x,u,r,t) %these are the Jacobians of the continuous function. %assuming r (and thus B) is constant over timestep
    global J
    w = x(1:3);
    q = x(end-3:end);
    [Rt1, Rt2, Rt3, Rt4] = Rti(q);
    B = magneticFieldECI(r,t)
    
    %[distTorqJacW,distTorqJacQ] = distTorqJac(r,x);
    jww = J\(skewsym(J*w) - skewsym(w)*J); %+ distTorqJacW);
    jwq = J\(skewsym(u)*[Rt1*B Rt2*B Rt3*B Rt4*B]);% distTorqJacQ);
    jqw =  0.5*Wmat(q);
    jqq = 0.5*[0 -w.';w -skewsym(w)];
    jacX = [jww jwq; jqw jqq];
    
    jwu = -J\skewsym(rotT(q)*B);
    jqu = zeros(4,3);
    jacU = [jwu;jqu];
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
function xkp1 = rk4(xk,uk,rk,tk)
    global dt
    k1 = dynamics(xk,uk,rk,tk);
    k2 = dynamics(xk+0.5*dt*k1,uk,rk,tk);
    k3 = dynamics(xk+0.5*dt*k2,uk,rk,tk);
    k4 = dynamics(xk+dt*k3,uk,rk,tk);
    xkp1 = xk + (dt/6)*(k1+2*k2+2*k3+k4);
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
function ce = eqConstraints(k,x,u)  %ineq before eq
    ce = [];
end
function c = constraints(k,x,u)
    c = [ineqConstraints(k,x,u);eqConstraints(k,x,u)];
end