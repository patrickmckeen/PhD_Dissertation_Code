global G M
G = 6.67408*10^-11;
M = 5.972*10^24;
%test for ~400 km equatorial orbit
%[r, rdot] = kep_ECI(0, 6771*1000, 0, 0, 0, 90, 180)
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