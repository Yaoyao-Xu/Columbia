% clc
% clear
% w=100;
% bck=zeros(28,3)*0.5;

% bck(:,3)=1000;
%%
% Define global variables:

function tdistance=Motion(w,bck)
global g
 g = -9.81; % in m/s^2 %%%%%%
global dt
 dt = .001;%%%%%%%%%%
global T;
 T = 0;

kg = 1000 ;% ground k
u = 0.5;% ground friction
%load bck
b = bck(1:28);
c = bck(29:56);
k = bck(57:84);
m = .8;% mass
%origin length
l = .1;
L0 = .1;
%cube
Vo = [0 0 0; 0 L0 0; L0 L0 0; L0 0 0; 0 0 L0; 0 L0 L0; L0 L0 L0; L0 0 L0;];
V = [Vo(:,1)+.1,Vo(:,2)+.1,Vo(:,3)+.1];
%initial condition
posx = V(:,1);
posy = V(:,2);
posz = V(:,3);
vx = zeros(length(posx),1);%
vy = zeros(length(posx),1);
vz = zeros(length(posx),1);
agx = zeros(length(posx),1);%
agy = zeros(length(posx),1);
agz = zeros(length(posx),1);
mcposx = NaN(1,3/dt);
mcposy = NaN(1,3/dt);
mcposz = NaN(1,3/dt);
%
n = 0;
while n<=3000
 n = n+1;
 l0 = -b.*sin(w*T+c) + b.*sin(w*(T+dt)+c);%%calculate l0
 V = [posx,posy,posz];%cube
 [Fspring,mo] = spring(l,V,k,l0);%get spring force and l0in xyz
 mox = mo(:,1);
 moy = mo(:,2);
 moz = mo(:,3);
 %calculate mmotion
 mtx=vx*dt + mox;
 mty=vy*dt + moy;
 mtz=vz*dt + moz;
 % get position
 posx = posx + mtx; %%
 posy = posy + mty;
 posz = posz + mtz;
 for i=1: length(posx) %% when touch ground
 if posz(i) <= 0
 agz(i)=-kg*posz(i)/m;
 if mtx(i)^2+mty(i)^2 ==0
 agx(i)=0;
 agy(i)=0;
 else
 agx(i)=kg*posz(i)/m*u*(mtx(i)/sqrt(mtx(i)^2+mty(i)^2));
 agy(i)=kg*posz(i)/m*u*(mty(i)/sqrt(mtx(i)^2+mty(i)^2));
 end
 else
 agz(i)=0;
 agx(i)=0;
 agy(i)=0;
 end
 end
 %spring force
 Fspringx = Fspring(:,1); %%
 Fspringy = Fspring(:,2);
 Fspringz = Fspring(:,3);

 %spring acceleration
 aspringx = Fspringx./m;
 aspringy = Fspringy./m;
 aspringz = Fspringz./m;
 % speed
 vx = vx + aspringx*dt+agx*dt;
 vy = vy + aspringy*dt+agy*dt;
 vz = vz + aspringz*dt+g*dt+agz*dt;%%
 % record position
 Distancex(:,n) = posx;
 Distancey(:,n) = posy;
 Distancez(:,n) = posz;
 T = T+dt;
end
tdistance = sqrt(mean(posx)^2 + mean(posy)^2);
end
