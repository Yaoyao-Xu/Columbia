function [tdistance] = Motion(w,bckmv)
%% debug
% clc
% clear
% w=100;
%
% m=ones(1,8)*0.1;
% b = zeros(1,28);
% c = zeros(1,28);
% k =1000*ones(1,28);
% L0=0.1;
% v=[0 0 L0 L0 0 0 L0 L0 0 L0 L0 0 0 L0 L0 0 0 0 0 0 L0 L0 L0 L0 ];
% bckmv=[b c k m v];
%% put gene in to data
b = bckmv(1:28);
c = bckmv(29:56);
k = bckmv(57:84);
m = bckmv(85:92)';
v1= bckmv(93:100)';
v2= bckmv(101:108)';
v3= bckmv(109:116)';
%% Define global variables:
global g
 g = -9.81; % in m/s^2 gravity
global dt
 dt = .001;% step time
global T; % Time
 T = 0;

kg = 50 ; % gorud coefficient
u = 0.5;% groung fridction coeficient
damp=0.29;% damping coeficient
Vo = [v1,v2,v3]; % initial positon of robot
V = [Vo(:,1),Vo(:,2),Vo(:,3)];
[l]=lengthorigin(Vo); % origin length of all spring
% set start condition
posx = V(:,1);
posy = V(:,2);
posz = V(:,3);
vx = zeros(length(posx),1);%
vy = zeros(length(posx),1);
vz = zeros(length(posx),1);
agx = zeros(length(posx),1);%
agy = zeros(length(posx),1);
agz = zeros(length(posx),1);
%
n = 0;
while n<= 3/dt
 n = n+1;
 l0 = -b.*sin(w*T+c) + b.*sin(w*(T+dt)+c);%% calulate motion from sine wave
 V = [posx,posy,posz];
 [Fspring,mo] = spring(l,V,k,l0); % divide spring force and sine motion in to masses
 mox = mo(:,1);
 moy = mo(:,2);
 moz = mo(:,3);
 % calcualte motion
 mtx=vx*dt + mox;
 mty=vy*dt + moy;
 mtz=vz*dt + moz;
 % calculate total motion
 posx = posx + mtx; %%
 posy = posy + mty;
 posz = posz + mtz;
 for i=1: length(posx) %when masses touch ground
 if posz(i) <= 0
 agz(i)=-kg*posz(i)/m(i);
 if mtx(i)^2+mty(i)^2 ==0% if masses have friciton
 agx(i)=0;
 agy(i)=0;
 else
 agx(i)=kg*posz(i)/m(i)*u*(mtx(i)/sqrt(mtx(i)^2+mty(i)^2));
 agy(i)=kg*posz(i)/m(i)*u*(mty(i)/sqrt(mtx(i)^2+mty(i)^2));
 end
 else
 agz(i)=0;
 agx(i)=0;
 agy(i)=0;
 end
 end
 Fspringx = Fspring(:,1);
 Fspringy = Fspring(:,2);
 Fspringz = Fspring(:,3);


 aspringx = Fspringx./m;
 aspringy = Fspringy./m;
 aspringz = Fspringz./m;


 % convert damping force to velocity change
 vdx=vx*damp.*abs(vx)./m*dt;
 vdy=vy*damp.*abs(vy)./m*dt;
 vdz=vz*damp.*abs(vz)./m*dt;

 % get the new velocity
 vx = vx + aspringx*dt+agx*dt-vdx;
 vy = vy + aspringy*dt+agy*dt-vdy;
 vz = vz + aspringz*dt+g*dt+agz*dt-vdz;%%

 T = T+dt;
end
tdistance = sqrt((mean(posx))^2 + (mean(posy))^2); % calculae motion
end