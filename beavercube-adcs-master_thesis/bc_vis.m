global satAlignVector1 satAlignVector2 ECIAlignVector1 ECIAlignVector2 umax
global passtime passdur proptime decaytime
N = 3600;
passdur = 480;
passtime = 1200;
proptime = 2400;
decaytime = 100;
unitf = @(x) x./vecnorm(x);
satAlignVector1vis = [0 0 -1].';
satAlignVector2vis = [1 0 0].';
alignVector1import = @(k) double(k<(passtime+passdur/2)) + (k>=(passtime+passdur/2)).*(k<proptime).*(1-((k-(passtime+passdur/2))/(proptime-(passtime+passdur/2))));
alignVector2import = @(k) double(k>=proptime) + (k>=(passtime+passdur/2)).*(k<proptime).*(((k-(passtime+passdur/2))/(proptime-(passtime+passdur/2))));
alignVector1use = @(k) (abs(k-passtime)<passdur/2);
alignVector2use = @(k) (k>=proptime);
ECIAlignVector1vis = @(k,rk,vk) unitf(6370*Rset(:,passtime)/norm(Rset(:,passtime)) - rk);
ECIAlignVector2vis = @(k,rk,vk) unitf(-vk./vecnorm(vk));
newData1 = load('-mat', 'complex3');

% Create new variables in the base workspace from those fields.
vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, newData1.(vars{i}));
end


hh = figure('Position',[100 200 1200 420]);

h0 = subplot(3,2,[1 3 5]);
axis([-0.3 0.3 -0.3 0.3 -0.4 0.2])
[sx,sy,sz] = sphere(40);
ss = [shiftdim(sx,-1);shiftdim(sy,-1);shiftdim(sz,-1)];
tmp = struct('cdata',[],'colormap',[]);
nodes = 3*[0.03 0.03 0.03 0.03 -0.03 -0.03 -0.03 -0.03; 0.01 0.01 -0.01 -0.01 0.01 0.01 -0.01 -0.01; 0.01 -0.01 0.01 -0.01 0.01 -0.01 0.01 -0.01]/2;

h0.ClippingStyle = 'rectangle';
sides = [1 2 4 3 0 0 0 0;
	     0 0 0 0 1 2 4 3;
		 1 2 0 0 4 3 0 0;
		 0 0 1 2 0 0 4 3;
		 1 0 2 0 4 0 3 0;
		 0 1 0 2 0 4 0 3;];
     
     
     
  
    
sscale = 2;
soffset = 1.05; 
cp = [-0.3 -0.6 0.1];
cva = 15;
ct = [0 0 -0.02];
tscale = 2;
uscale = 3;






figure(hh)

% % 
h1 = subplot(3,2,2);
% x1 = xlabel('Time (s)');
hold off
plot(angdiff.')
a1 = axis;
hold on
% % plot([k k], a1(3:4), 'k--')
plot((passtime-passdur/2)*[1 1], a1(3:4), 'r--')
plot((passtime+passdur/2)*[1 1], a1(3:4), 'r--')
plot((proptime)*[1 1], a1(3:4), 'r--')
y1 = ylabel('Pointing Err (deg)','Interpreter','latex');
t1a = text((passtime-passdur/2)/2,[0.2 0.8]*a1(3:4).','Free','HorizontalAlignment','center','Interpreter','latex');
t1b = text(passtime,[0.2 0.8]*a1(3:4).','Goal 1','HorizontalAlignment','center','Interpreter','latex');
t1c = text(passtime*0.5 + proptime*0.5,[0.2 0.8]*a1(3:4).','Free','HorizontalAlignment','center','Interpreter','latex');
t1d = text(N*0.5 + proptime*0.5,[0.2 0.8]*a1(3:4).','Goal 2','HorizontalAlignment','center','Interpreter','latex');
axis(a1)
% % 
h2 = subplot(3,2,4);
% % x2 = xlabel('Time (s)');
hold off
plot((180.0/pi)*Xset(1:3,:).')
a2 = axis;
% % hold on
% % plot([k k], a2(3:4), 'k--')
plot((passtime-passdur/2)*[1 1], a2(3:4), 'r--')
plot((passtime+passdur/2)*[1 1], a2(3:4), 'r--')
plot((proptime)*[1 1], a2(3:4), 'r--')
y2 = ylabel('Ang Vel (deg/s)','Interpreter','latex');
axis(a2)
% % 
h3 = subplot(3,2,6);
hold off
plot(Uset.')
a3 = axis;
hold on
% % plot([k k], a3(3:4), 'k--')
plot((passtime-passdur/2)*[1 1], a3(3:4), 'r--')
plot((passtime+passdur/2)*[1 1], a3(3:4), 'r--')
plot((proptime)*[1 1], a3(3:4), 'r--')
x3 = xlabel('Time (s)','Interpreter','latex');
y3 = ylabel('Mag Moment (Am$^2$)','Interpreter','latex');
axis(a3)
uistack(h1,'top')
uistack(h2,'top')
uistack(h3,'top')
y1.Position(1) = -310;
y2.Position(1) = -310;
y3.Position(1) = -310;

% 




hp = figure('Position',[100 200 700 420]);

% % 
h1 = subplot(3,1,1);
% x1 = xlabel('Time (s)');
hold off
plot(angdiff.')
a1 = axis;
hold on
% % plot([k k], a1(3:4), 'k--')
plot((passtime-passdur/2)*[1 1], a1(3:4), 'r--')
plot((passtime+passdur/2)*[1 1], a1(3:4), 'r--')
plot((proptime)*[1 1], a1(3:4), 'r--')
y1 = ylabel('Pointing Err (deg)','Interpreter','latex');
t1a = text((passtime-passdur/2)/2,[0.2 0.8]*a1(3:4).','Free','HorizontalAlignment','center','Interpreter','latex');
t1b = text(passtime,[0.2 0.8]*a1(3:4).','Goal 1','HorizontalAlignment','center','Interpreter','latex');
t1c = text(passtime*0.5 + proptime*0.5,[0.2 0.8]*a1(3:4).','Free','HorizontalAlignment','center','Interpreter','latex');
t1d = text(N*0.5 + proptime*0.5,[0.2 0.8]*a1(3:4).','Goal 2','HorizontalAlignment','center','Interpreter','latex');
axis(a1)
% % 
h2 = subplot(3,1,2);
% % x2 = xlabel('Time (s)');
hold off
plot((180.0/pi)*Xset(1:3,:).')
a2 = axis;
hold on
% % plot([k k], a2(3:4), 'k--')
plot((passtime-passdur/2)*[1 1], a2(3:4), 'r--')
plot((passtime+passdur/2)*[1 1], a2(3:4), 'r--')
plot((proptime)*[1 1], a2(3:4), 'r--')
y2 = ylabel('Ang Vel (deg/s)','Interpreter','latex');
axis(a2)
% % 
h3 = subplot(3,1,3);
hold off
plot(Uset.')
a3 = axis;
hold on
% % plot([k k], a3(3:4), 'k--')
plot((passtime-passdur/2)*[1 1], a3(3:4), 'r--')
plot((passtime+passdur/2)*[1 1], a3(3:4), 'r--')
plot((proptime)*[1 1], a3(3:4), 'r--')
x3 = xlabel('Time (s)','Interpreter','latex');
y3 = ylabel('Mag Moment (Am$^2$)','Interpreter','latex');
axis(a3)
uistack(h1,'top')
uistack(h2,'top')
uistack(h3,'top')
y1.Position(1) = -310;
y2.Position(1) = -310;
y3.Position(1) = -310;

% 
saveas(hp,'anim_plots.png'); 


filename = 'traj_anim5.gif';
filename0 = 'traj_anim5_start.png';

for k = 1:10:(N-1)
%     figure(hh)
	xk = Xset(:,k);
	xdesk = Xdesset(:,k);
	q = xk(end-3:end);
	rk = Rset(:,k);
	vk = Vset(:,k);
	qframe = qcommand_PM(rk,vk,[0;0;1],[0;1;0]);
	Rframe = rot(qframe);
	Rsat = rotT(q);
	
    
    figure(hh)
    h0 = subplot(3,2,[1 3 5]);
    h0.ClippingStyle = 'rectangle';
    axis equal
    campos(cp)
    camva(cva)
    grid off
    
% 	rotsphere = sscale*mtimesx(jroty(2*k*2*pi/N),ss);
	rotsphere = sscale*mtimesx(Rframe,ss);
    hold off
	surf(squeeze(rotsphere(1,:,:)),squeeze(rotsphere(2,:,:)),squeeze(rotsphere(3,:,:))-sscale*soffset,'EdgeColor','g','FaceColor','w')
	
    axis equal
    campos(cp)
    camva(cva)
    camtarget(ct)

% 	campos([-0.5 -0.5 -0.1])
%     camva(30)
    hold on
	
	
	
	%pointing colors
    col1base = [1 0 0];
    col2base = [1 0 0];
    col1 = col1base + ([1 1 1] - col1base)*(1-alignVector1import(k));
    col2 = col2base + ([1 1 1] - col2base)*(1-alignVector2import(k));
    if alignVector1use(k)
        ls1 = '-';
        dls1 = ':';
    else
        ls1 = '-';
        dls1 = ':';
    end
    if alignVector2use(k)
        ls2 = '-';
        dls2 = ':';
    else
        ls2 = '-';
        dls2 = ':';
    end
    
    %draw target
%     targ = rk - Rset(:,k)/norm(Rset(:,k));
    targ = sscale*Rset(:,passtime)/norm(Rset(:,passtime));
    rottarg = Rframe*targ - sscale*soffset*[0;0;1]; 
    tline = plot3(rottarg(1),rottarg(2),rottarg(3),'m*','MarkerSize',16);
    
    lw = 2;
	%draw pointing
	p1 = Rframe*Rsat.'*((satAlignVector1vis)*[0.005 0.075]);
	p2 = Rframe*Rsat.'*((satAlignVector2vis)*[0.005 0.075]);
    if k<=passtime+passdur/2
        pline = plot3(p1(1,:),p1(2,:),p1(3,:),'Color',col1base,'LineStyle',ls1,'LineWidth',lw );
    end
    if k>passtime+passdur/2
        pline = plot3(p2(1,:),p2(2,:),p2(3,:),'Color',col2base,'LineStyle',ls2,'LineWidth',lw );
    end
        
	% draw desired pointing
	d1 = Rframe*((ECIAlignVector1vis(k,rk,vk))*[0.005 0.15]);
    d1p = [[0;0;0] rottarg];
	d2 = Rframe*((ECIAlignVector2vis(k,rk,vk))*[0.005 0.15]);
    if ~alignVector2use(k)
        d1line = plot3(d1(1,:),d1(2,:),d1(3,:),'Color',col1,'LineStyle',dls1,'LineWidth',lw );
%         plot3(d1(1,:),d1(2,:),d1(3,:),'Color',col1,'MarkerStyle','o','LineWidth',lw )
%         plot3(d1p(1,:),d1p(2,:),d1p(3,:),'m:');
    end
    if k>passtime+passdur/2
        d2line = plot3(d2(1,:),d2(2,:),d2(3,:),'Color',col2,'LineStyle',dls2,'LineWidth',lw );
    end
    
    
    %draw magnetic field
    bk = Bset(:,k);
    B = Rframe*(bk/norm(bk))*[0.05 -0.05];
    mline = plot3(B(1,:),B(2,:),B(3,:),'b-.','LineWidth',2);
    
    %draw magnetic moment
    uk = Rsat.'*Uset(:,k)/0.2;
    urot = Rframe*(uk*uscale)*[0.05 -0.05];
    uline = plot3(urot(1,:),urot(2,:),urot(3,:),'g-.','LineWidth',2);
    
    %draw torques
    tqk = Rframe*cross(bk,uk)*tscale;
    tqkplot = (tqk/norm(bk))*[0 0.05];
%     from = Rframe*Rsat.'*(
    tqline = plot3(tqkplot(1,:),tqkplot(2,:),tqkplot(3,:),'c-','LineWidth',lw);
%     createTorqueArrow3((tqk/norm(tqk)),[0;0;0],0.02,B(:,2),0.01)
    
	%draw body
	b = Rframe*Rsat.'*nodes;
	for j = 1:6
		st = b(:,logical(sides(j,:)));
        order = [1 2 4 3 1];
		plot3(st(1,order),st(2,order),st(3,order),'black')
% 		patch(st(1,order),st(2,order),st(3,order),'yellow')
	end
    
    axis equal
    campos(cp)
    camtarget(ct)
    camva(cva)
    grid off
    if ~alignVector2use(k) && k>passtime+passdur/2
        legend([tline,pline,d1line,d2line,mline,uline,tqline],'Target','Pointing Axis','Pointing Goal 1 (x)','Pointing Goal 2 (z)','Earth Mag Field','Control Mag Moment','Control Torque')
%         plot3(d1p(1,:),d1p(2,:),d1p(3,:),'m:');
    elseif ~alignVector2use(k)
        legend([tline,pline,d1line,mline,uline,tqline],'Target','Pointing Axis','Pointing Goal 1 (x)','Earth Mag Field','Control Mag Moment','Control Torque')
    elseif k>passtime+passdur/2
        legend([tline,pline,d2line,mline,uline,tqline],'Target','Pointing Axis','Pointing Goal 2 (z)','Earth Mag Field','Control Mag Moment','Control Torque')
    end
    
    
    h1 = subplot(3,2,2);
    % x1 = xlabel('Time (s)');
    hold off
    plot(angdiff.')
    hold on
    plot([k k], a1(3:4), 'k--')
    plot((passtime-passdur/2)*[1 1], a1(3:4), 'r--')
    plot((passtime+passdur/2)*[1 1], a1(3:4), 'r--')
    plot((proptime)*[1 1], a1(3:4), 'r--')
    y1 = ylabel('Pointing Err (deg)');
    t1a = text((passtime-passdur/2)/2,[0.2 0.8]*a1(3:4).','Free','HorizontalAlignment','center','Interpreter','latex');
    t1b = text(passtime,[0.2 0.8]*a1(3:4).','Goal 1','HorizontalAlignment','center','Interpreter','latex');
    t1c = text((passtime+passdur/2)*0.5 + proptime*0.5,[0.2 0.8]*a1(3:4).','Free','HorizontalAlignment','center','Interpreter','latex');
    t1d = text(N*0.5 + proptime*0.5,[0.2 0.8]*a1(3:4).','Goal 2','HorizontalAlignment','center','Interpreter','latex');
    axis(a1);

    h2 = subplot(3,2,4);
    % x2 = xlabel('Time (s)');
    hold off
    plot((180.0/pi)*Xset(1:3,:).')
    hold on
    plot([k k], a2(3:4), 'k--')
    plot((passtime-passdur/2)*[1 1], a2(3:4), 'r--')
    plot((passtime+passdur/2)*[1 1], a2(3:4), 'r--')
    plot((proptime)*[1 1], a2(3:4), 'r--')
    y2 = ylabel('Ang Vel (deg/s)');
    axis(a2);

    h3 = subplot(3,2,6);
    hold off
    plot(Uset.')
    hold on
    plot([k k], a3(3:4), 'k--')
    plot((passtime-passdur/2)*[1 1], a3(3:4), 'r--')
    plot((passtime+passdur/2)*[1 1], a3(3:4), 'r--')
    plot((proptime)*[1 1], a3(3:4), 'r--')
    x3 = xlabel('Time (s)');
    y3 = ylabel('Mag Moment (Am2)');
    axis(a3);
    y1.Position(1) = -310;
    y2.Position(1) = -310;
    y3.Position(1) = -310;
    
    uistack(h1,'top')
    uistack(h2,'top')
    uistack(h3,'top')
%     uistack(y1,'top')
%     uistack(y2,'top')
%     uistack(y3,'top')
%     uistack(x3,'top')


    drawnow
%     M(k) = getframe;
    frame = getframe(hh); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    % Write to the GIF File 
    if k == 1 
      imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      saveas(hh,filename0); 

    else 
      imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0.1); 
    end 
    if mod(k-1,50) == 0
        ht = figure('Position',[100 200 700 420]);    
    % 	rotsphere = sscale*mtimesx(jroty(2*k*2*pi/N),ss);
        rotsphere = sscale*mtimesx(Rframe,ss);
        hold off
        surf(squeeze(rotsphere(1,:,:)),squeeze(rotsphere(2,:,:)),squeeze(rotsphere(3,:,:))-sscale*soffset,'EdgeColor','g','FaceColor','w')

        axis equal
        campos(cp)
        camva(cva)
        camtarget(ct)

    % 	campos([-0.5 -0.5 -0.1])
    %     camva(30)
        hold on



        %pointing colors
        col1base = [1 0 0];
        col2base = [1 0 0];
        col1 = col1base + ([1 1 1] - col1base)*(1-alignVector1import(k));
        col2 = col2base + ([1 1 1] - col2base)*(1-alignVector2import(k));
        if alignVector1use(k)
            ls1 = '-';
            dls1 = ':';
        else
            ls1 = '-';
            dls1 = ':';
        end
        if alignVector2use(k)
            ls2 = '-';
            dls2 = ':';
        else
            ls2 = '-';
            dls2 = ':';
        end

        %draw target
    %     targ = rk - Rset(:,k)/norm(Rset(:,k));
        targ = sscale*Rset(:,passtime)/norm(Rset(:,passtime));
        rottarg = Rframe*targ - sscale*soffset*[0;0;1]; 
        tline = plot3(rottarg(1),rottarg(2),rottarg(3),'m*','MarkerSize',16);

        lw = 2;
        %draw pointing
        p1 = Rframe*Rsat.'*((satAlignVector1vis)*[0.005 0.075]);
        p2 = Rframe*Rsat.'*((satAlignVector2vis)*[0.005 0.075]);
        if k<=passtime+passdur/2
            pline = plot3(p1(1,:),p1(2,:),p1(3,:),'Color',col1base,'LineStyle',ls1,'LineWidth',lw );
        end
        if k>passtime+passdur/2
            pline = plot3(p2(1,:),p2(2,:),p2(3,:),'Color',col2base,'LineStyle',ls2,'LineWidth',lw );
        end

        % draw desired pointing
        d1 = Rframe*((ECIAlignVector1vis(k,rk,vk))*[0.005 0.1]);
        d1p = [[0;0;0] rottarg];
        d2 = Rframe*((ECIAlignVector2vis(k,rk,vk))*[0.005 0.1]);
        if ~alignVector2use(k)
            d1line = plot3(d1(1,:),d1(2,:),d1(3,:),'Color',col1,'LineStyle',dls1,'LineWidth',lw );
    %         plot3(d1(1,:),d1(2,:),d1(3,:),'Color',col1,'MarkerStyle','o','LineWidth',lw )
    %         plot3(d1p(1,:),d1p(2,:),d1p(3,:),'m:');
        end
        if k>passtime+passdur/2
            d2line = plot3(d2(1,:),d2(2,:),d2(3,:),'Color',col2,'LineStyle',dls2,'LineWidth',lw );
        end


        %draw magnetic field
        bk = Bset(:,k);
        B = Rframe*(bk/norm(bk))*[0.1 -0.1];
        mline = plot3(B(1,:),B(2,:),B(3,:),'b-.','LineWidth',2);

        %draw magnetic moment
        uk = Rsat.'*Uset(:,k)/0.2;
        urot = Rframe*(uk*uscale)*[0.05 -0.05];
        uline = plot3(urot(1,:),urot(2,:),urot(3,:),'g-.','LineWidth',2);

        %draw torques
        tqk = Rframe*cross(bk,uk)*tscale;
        tqkplot = (tqk/norm(bk))*[0 0.05];
    %     from = Rframe*Rsat.'*(
        tqline = plot3(tqkplot(1,:),tqkplot(2,:),tqkplot(3,:),'c-','LineWidth',lw);
    %     createTorqueArrow3((tqk/norm(tqk)),[0;0;0],0.02,B(:,2),0.01)

        %draw body
        b = Rframe*Rsat.'*nodes;
        for j = 1:6
            st = b(:,logical(sides(j,:)));
            order = [1 2 4 3 1];
            plot3(st(1,order),st(2,order),st(3,order),'black')
    % 		patch(st(1,order),st(2,order),st(3,order),'yellow')
        end

        axis equal
        campos(cp)
        camtarget(ct)
        camva(cva)
        grid off
        if ~alignVector2use(k) && k>passtime+passdur/2
            legend([tline,pline,d1line,d2line,mline,uline,tqline],'Target','Pointing Axis','Pointing Goal 1 (x)','Pointing Goal 2 (z)','Earth Mag Field','Control Mag Moment','Control Torque')
    %         plot3(d1p(1,:),d1p(2,:),d1p(3,:),'m:');
        elseif ~alignVector2use(k)
            legend([tline,pline,d1line,mline,uline,tqline],'Target','Pointing Axis','Pointing Goal 1 (x)','Earth Mag Field','Control Mag Moment','Control Torque')
        elseif k>passtime+passdur/2
            legend([tline,pline,d2line,mline,uline,tqline],'Target','Pointing Axis','Pointing Goal 2 (z)','Earth Mag Field','Control Mag Moment','Control Torque')
        end
        drawnow
        saveas(ht,['anim_stills/traj_anim_still_',int2str(k),'.png']);
        close(ht);
    end
   
end
%%

sscale = 2;
soffset = 1.05; 
cp = [-0.3 -0.6 0.1];
cva = 11;
ct = [0 0 -0.02];
tscale = 2;
uscale = 3;
hs = figure('Position',[100 200 400 200]);
img = [];
c = 0;

klist = [1,3599,3300,3000,2700,2400,...
            2500,2390,2200,2000,1800,...
            1500,980,1100,1147,1150,1175,1200,1225,1250,1300,...
            1400,100,300,400,500,800];
            
klist = sort(klist);

for k = klist
%     figure(hh)
	xk = Xset(:,k);
	xdesk = Xdesset(:,k);
	q = xk(end-3:end);
	rk = Rset(:,k);
	vk = Vset(:,k);
	qframe = qcommand_PM(rk,vk,[0;0;1],[0;1;0]);
	Rframe = rot(qframe);
	Rsat = rotT(q);
    c = c+1;
	
    
    figure(hs)
    h0 = subplot(3,2,[1 3 5]);
    h0.ClippingStyle = 'rectangle';
    axis equal
    campos(cp)
    camva(cva)
    grid off
    
% 	rotsphere = sscale*mtimesx(jroty(2*k*2*pi/N),ss);
	rotsphere = sscale*mtimesx(Rframe,ss);
    hold off
	surf(squeeze(rotsphere(1,:,:)),squeeze(rotsphere(2,:,:)),squeeze(rotsphere(3,:,:))-sscale*soffset,'EdgeColor','g','FaceColor','w')
	
    axis equal
    campos(cp)
    camva(cva)
    camtarget(ct)

% 	campos([-0.5 -0.5 -0.1])
%     camva(30)
    hold on
	
	
	
	%pointing colors
    col1base = [1 0 0];
    col2base = [1 0 0];
    col1 = col1base + ([1 1 1] - col1base)*(1-alignVector1import(k));
    col2 = col2base + ([1 1 1] - col2base)*(1-alignVector2import(k));
    if alignVector1use(k)
        ls1 = '-';
        dls1 = ':';
    else
        ls1 = '-';
        dls1 = ':';
    end
    if alignVector2use(k)
        ls2 = '-';
        dls2 = ':';
    else
        ls2 = '-';
        dls2 = ':';
    end
    
    %draw target
%     targ = rk - Rset(:,k)/norm(Rset(:,k));
    targ = sscale*Rset(:,passtime)/norm(Rset(:,passtime));
    rottarg = Rframe*targ - sscale*soffset*[0;0;1]; 
    tline = plot3(rottarg(1),rottarg(2),rottarg(3),'m*','MarkerSize',16);
    
    lw = 2;
	%draw pointing
	p1 = Rframe*Rsat.'*((satAlignVector1vis)*[0.005 0.075]);
	p2 = Rframe*Rsat.'*((satAlignVector2vis)*[0.005 0.075]);
    if k<=passtime+passdur/2
        pline = plot3(p1(1,:),p1(2,:),p1(3,:),'Color',col1base,'LineStyle',ls1,'LineWidth',lw );
    end
    if k>passtime+passdur/2
        pline = plot3(p2(1,:),p2(2,:),p2(3,:),'Color',col2base,'LineStyle',ls2,'LineWidth',lw );
    end
        
	% draw desired pointing
	d1 = Rframe*((ECIAlignVector1vis(k,rk,vk))*[0.005 0.1]);
    d1p = [[0;0;0] rottarg];
	d2 = Rframe*((ECIAlignVector2vis(k,rk,vk))*[0.005 0.1]);
    if ~alignVector2use(k)
        d1line = plot3(d1(1,:),d1(2,:),d1(3,:),'Color',col1,'LineStyle',dls1,'LineWidth',lw );
%         plot3(d1(1,:),d1(2,:),d1(3,:),'Color',col1,'MarkerStyle','o','LineWidth',lw )
%         plot3(d1p(1,:),d1p(2,:),d1p(3,:),'m:');
    end
    if k>passtime+passdur/2
        d2line = plot3(d2(1,:),d2(2,:),d2(3,:),'Color',col2,'LineStyle',dls2,'LineWidth',lw );
    end
    
    
    %draw magnetic field
    bk = Bset(:,k);
    B = Rframe*(bk/norm(bk))*[0.1 -0.1];
    mline = plot3(B(1,:),B(2,:),B(3,:),'b-.','LineWidth',2);
    
    %draw magnetic moment
    uk = Rsat.'*Uset(:,k)/0.2;
    urot = Rframe*(uk*uscale)*[0.05 -0.05];
    uline = plot3(urot(1,:),urot(2,:),urot(3,:),'g-.','LineWidth',2);
    
    %draw torques
    tqk = Rframe*cross(bk,uk)*tscale;
    tqkplot = (tqk/norm(bk))*[0 0.05];
%     from = Rframe*Rsat.'*(
    tqline = plot3(tqkplot(1,:),tqkplot(2,:),tqkplot(3,:),'c-','LineWidth',lw);
%     createTorqueArrow3((tqk/norm(tqk)),[0;0;0],0.02,B(:,2),0.01)
    
	%draw body
	b = Rframe*Rsat.'*nodes;
	for j = 1:6
		st = b(:,logical(sides(j,:)));
        order = [1 2 4 3 1];
		plot3(st(1,order),st(2,order),st(3,order),'black')
% 		patch(st(1,order),st(2,order),st(3,order),'yellow')
	end
    if c == 1
%         camproj('perspective');
        set(gca, 'CameraPositionMode', 'manual', ...
         'CameraTargetMode', 'manual', ...
         'CameraUpVectorMode', 'manual', ...
         'CameraViewAngleMode', 'manual');

        ll = legend([tline,pline,d1line,mline,uline,tqline],'Target','Pointing Axis','Pointing Goal(s)','Earth Mag Field','Control Mag Moment','Control Torque')
        oldpos = ll.Position
        ll.Position = [0.6,0.2,oldpos(3:end)];
%         plot3(d1p(1,:),d1p(2,:),d1p(3,:),'m:');
    end
    drawnow;
    
    axis equal
    campos(cp)
    camtarget(ct)
    camva(cva)
    grid off
    annotation('textbox',[0.75, 0.8,0.2,0.1],'String', ['t = ',int2str(k)], 'FitBoxToText', 'on', 'Color', 'black', 'BackgroundColor', 'white');

    
    
    drawnow
%     M(k) = getframe;
    frame = getframe(hs); 
    im = frame2im(frame); 
%     [imind,cm] = rgb2ind(im,256); 
    % Write to the GIF File 
    img = [img;im]; 
%     if k == 1 
%       imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
%     else 
%       imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0.1); 
%     end 
   
end
imh = size(img,1)/c;
img2 = [img(1:imh*c/3,:,:), img(imh*c/3 + 1:2*imh*c/3,:,:), img(2*imh*c/3 + 1:end,:,:)];
ff = figure;
imshow(img2)
print('anim_stack','-dtiff','-r1200')

% M(end+1) = tmp;
% save('bc_anim','M')
% movie(M)

%https://www.mathworks.com/matlabcentral/answers/94495-how-can-i-create-animated-gif-images-in-matlab
% timestep = 1;
% h = figure;
% axis tight manual % this ensures that getframe() returns a consistent size
% filename = 'testAnimated.gif';
% for n = 1:timestep:N-1
%     % Draw plot for y = x.^n
%     
%     plot(x,y) 
%     drawnow 
%       % Capture the plot as an image 
%     frame = getframe(h); 
%     im = frame2im(frame); 
%     [imind,cm] = rgb2ind(im,256); 
%     % Write to the GIF File 
%     if n == 1 
%       imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
%     else 
%       imwrite(imind,cm,filename,'gif','WriteMode','append'); 
%     end 
%     end

%%



function Rx = jrotx(angle)
    angle = shiftdim(angle,-2);
    Rx = [ones(size(angle)) zeros(size(angle)) zeros(size(angle));
            zeros(size(angle)) cos(angle) -sin(angle);
            zeros(size(angle)) sin(angle) cos(angle)];
end

function Rx = jroty(angle)
    angle = shiftdim(angle,-2);
    Rx = [cos(angle)  zeros(size(angle)) -sin(angle);
            zeros(size(angle)) ones(size(angle))  zeros(size(angle));
             sin(angle) zeros(size(angle)) cos(angle)];
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
