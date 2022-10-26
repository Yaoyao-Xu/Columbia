clc
clear
load( 'data.mat')
bck=bck(1,:);
%% insert the motion part
% video part is just directly changed from motion
global g
 g = -9.81; % in m/s^2
global dt
 dt = .001;
global T;
 T = 0;

kg = 1000 ;% ground k
u = 0.5;% friction
%load bck
b = bck(1:28);
c = bck(29:56);
k = bck(57:84);
% mass
m = .8;
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
 V = [posx,posy,posz]; %cube
 [Fspring,mo] = spring(l,V,k,l0);%get spring force and l0in xyz
 mox = mo(:,1);
 moy = mo(:,2);
 moz = mo(:,3);
 %calculate mmotion
 mtx=vx*dt + mox;
 mty=vy*dt + moy;
 mtz=vz*dt + moz;
 % get position
 posx = posx + mtx;
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
%% Plot part
ccc=0;
E=[ 1 2;1 3;1 4;1 5;
 1 6;1 7;1 8;2 3;
 2 4;2 5;2 6;2 7;
 2 8;3 4;3 5;3 6;
 3 7;3 8;4 5;4 6;
 4 7;4 8;5 6;5 7;
 5 8;6 7;6 8;7 8]; %spring link
for loop=10:10:3000
 V=[Distancex(:,loop) Distancey(:,loop) Distancez(:,loop)]; %cube
figure(1) %plot cube
scatter3(V(:,1),V(:,2),V(:,3));
hold on;
for i=1:size(E,1)
V1=V(E(i,1),:);
V2=V(E(i,2),:);
line([V1(1) V2(1)],[V1(2) V2(2)],[V1(3) V2(3)]);
end
axis equal
axis([-5 5 -5 5 -1 5])%set frame size ,can change 5 to 1 to get more detail video
hold off
ccc=ccc+1;
M(ccc)=getframe;
end
ylabel('Meter','FontSize',14)
xlabel('Meter','FontSize',14)
zlabel('Meter','FontSize',14)
%title('Wire Frame Cube','FontSize',14)
figure(2)
axis off
movie(M,1,100)

