clearvars
close all
clc
w=pi;
global g
 g = -9.81; % in m/s^2 %%%%%%
global dt
 dt = .001;%%%%%%%%%%
global T;
 T = 0;
for a = 1:3
m=ones(1,8)*0.1;
b = 0.01*ones(1,28);
c = zeros(1,28);
k = 50*ones(1,28);
L0=1;
v=[0 0 L0 L0 0 0 L0 L0 0 L0 L0 0 0 L0 L0 0 0 0 0 0 L0 L0 L0 L0 ];
bckmv=[b c k m v];
filename = strcat('data_', num2str(a), '_2500.mat'); % load three sets of final data
load(filename);
% Best: data_1_2500, data6_3_2500
bckmv=bckmv(1,:);
%%
b = bckmv(1:28);
c = bckmv(29:56);
k = bckmv(57:84);
m = bckmv(85:92)';
v1= bckmv(93:100)';
v2= bckmv(101:108)';
v3= bckmv(109:116)';
%% Define variables:

kg = 50 ;%% %%%%%%%%%%%
u = 0.5;%%
damp=0.29;
L0 = .1;
Vo = [v1,v2,v3];
V = [Vo(:,1),Vo(:,2),Vo(:,3)];
[l]=lengthorigin(Vo);
if a == 1
 [idx_k0_1] = find(k==0); % find the index when k = 0
end
if a == 2
 [idx_k0_2] = find(k==0);
end
if a == 3
 [idx_k0_3] = find(k==0);
end
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
while n<= 10/dt
 n = n+1;
 l0 = -b.*sin(w*T+c) + b.*sin(w*(T+dt)+c);%%
 V = [posx,posy,posz];
 [Fspring,mo] = spring(l,V,k,l0);
 mox = mo(:,1);
 moy = mo(:,2);
 moz = mo(:,3);

 mtx=vx*dt + mox;
 mty=vy*dt + moy;
 mtz=vz*dt + moz;

 posx = posx + mtx; %%
 posy = posy + mty;
 posz = posz + mtz;
 for i=1: length(posx) %%%%%%%
 if posz(i) <= 0
 agz(i)=-kg*posz(i)/m(i);
 if mtx(i)^2+mty(i)^2 ==0
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
 Fspringx = Fspring(:,1); %%
 Fspringy = Fspring(:,2);
 Fspringz = Fspring(:,3);


 aspringx = Fspringx./m;
 aspringy = Fspringy./m;
 aspringz = Fspringz./m;



 vdx=vx*damp.*abs(vx)./m*dt;
 vdy=vy*damp.*abs(vy)./m*dt;
 vdz=vz*damp.*abs(vz)./m*dt;


 vx = vx + aspringx*dt+agx*dt-vdx;
 vy = vy + aspringy*dt+agy*dt-vdy;
 vz = vz + aspringz*dt+g*dt+agz*dt-vdz;%%
%
 Distancex(:,n) = posx;
 Distancey(:,n) = posy;
 Distancez(:,n) = posz;
%
 if a == 1 % store Distance data for different runs
 Distancex1 = Distancex;
 Distancey1 = Distancey;
 Distancez1 = Distancez;
 end
 if a == 2
 Distancex2 = Distancex;
 Distancey2 = Distancey;
 Distancez2 = Distancez;
 end
 if a == 3
 Distancex3 = Distancex;
 Distancey3 = Distancey;
 Distancez3 = Distancez;
 end



 T = T+dt;
end
tdistance = sqrt((mean(posx))^2 + (mean(posy))^2);
end
%% Plot part
ccc=0;
E=[ 1 4;2 3;5 8;6 7; %%%%%%%%%%%%%
 1 2;3 4;5 6;7 8;
 1 5;2 6;3 7;4 8;
 1 3;2 4;5 7;6 8;
 1 8;4 5;2 7;3 6;
 1 6;2 5;3 8;4 7;
 1 7;3 5;2 8;4 6];
E1 = E;
E2 = E;
E3 = E;
E1(idx_k0_1,:) = []; % delete the springs which have k = 0
E2(idx_k0_2,:) = [];
E3(idx_k0_3,:) = [];
for loop=10:10:10000

 VV1 = [Distancex1(:,loop) Distancey1(:,loop) Distancez1(:,loop)];
 VV2 = [Distancex2(:,loop)+5 Distancey2(:,loop) Distancez2(:,loop)];
 VV3 = [Distancex3(:,loop)+10 Distancey3(:,loop) Distancez3(:,loop)];

 % Plot three sets of mass points:
 figure(1)
 plot3(VV1(:,1),VV1(:,2),VV1(:,3),'.','Markersize',20,'color',[0.4940 0.1840 0.5560]);
 hold on
 plot3(VV2(:,1),VV2(:,2),VV2(:,3),'.','Markersize',20,'color',[0.4660 0.6740 0.1880]);
 plot3(VV3(:,1),VV3(:,2),VV3(:,3),'.','Markersize',20,'color',[0.8500 0.3250 0.0980]);

for i=1:size(E1,1)
V1=VV1(E1(i,1),:);
V2=VV1(E1(i,2),:);
line([V1(1) V2(1)],[V1(2) V2(2)],[V1(3) V2(3)],'LineWidth',1.1,'color',[0.9290 0.6940 0.1250]);
end
for ii = 1:size(E2,1)
V3=VV2(E2(ii,1),:);
V4=VV2(E2(ii,2),:);
line([V3(1) V4(1)],[V3(2) V4(2)],[V3(3) V4(3)],'LineWidth',1.1,'color','k');
end
for iii = 1:size(E3,1)
V5=VV3(E3(iii,1),:);
V6=VV3(E3(iii,2),:);
line([V5(1) V6(1)],[V5(2) V6(2)],[V5(3) V6(3)],'LineWidth',1.1);
end
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
axis equal
axis([-1 20 -1 15 -1 3])
set(gca,'XDir','reverse')
set(gca,'YDir','reverse')
grid minor
hold off
ccc=ccc+1;
M(ccc)=getframe;
end
ylabel('Meter','FontSize',14)
xlabel('Meter','FontSize',14)
zlabel('Meter','FontSize',14)
title('Wire Frame Cube','FontSize',14)
figure(2)
axis off
movie(M,1,100)
