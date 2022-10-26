clc
clear
% video part is just directly changed from motion
load( 'data.mat')
bckl=bck;

bck=bckl(1,:);
%% cube1 motion part
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
Dx1=Distancex;
Dy1=Distancey;
Dz1=Distancez;
%% cube2 motion part
bck=bckl(20,:);
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
Dx2=Distancex;
Dy2=Distancey;
Dz2=Distancez;
%% Plot part
ccc=0;
E=[ 1 2;1 3;1 4;1 5;
 1 6;1 7;1 8;2 3;
 2 4;2 5;2 6;2 7;
 2 8;3 4;3 5;3 6;
 3 7;3 8;4 5;4 6;
 4 7;4 8;5 6;5 7;
 5 8;6 7;6 8;7 8]; %%spring linkk
for loop=10:10:3000
 V11=[Dx1(:,loop) Dy1(:,loop) Dz1(:,loop)];
 V22=[Dx2(:,loop) Dy2(:,loop) Dz2(:,loop)];
figure(1)
scatter3(V11(:,1),V11(:,2),V11(:,3)); %%plot two cube
hold on;
for i=1:size(E,1)
V1=V11(E(i,1),:);
V2=V11(E(i,2),:);
line([V1(1) V2(1)],[V1(2) V2(2)],[V1(3) V2(3)]);
end
hold on
scatter3(V22(:,1),V22(:,2),V22(:,3));
hold on;
for i=1:size(E,1)
V1s=V22(E(i,1),:);
V2s=V22(E(i,2),:);
line([V1s(1) V2s(1)],[V1s(2) V2s(2)],[V1s(3) V2s(3)]);
end
axis equal
axis([-5 5 -5 5 0 5])%set frame size ,can set to 1 to get more detail video
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



