clc
clear
w=pi;
m=ones(1,8)*0.1;
b = 0.01*ones(1,28);
c = zeros(1,28);
k = 50*ones(1,28);
L0=1;
v=[0 0 L0 L0 0 0 L0 L0 0 L0 L0 0 0 L0 L0 0 0 0 0 0 L0 L0 L0 L0 ];
bckmv=[b c k m v];
load('result2.mat')% change to data_1_2500, data_3_2500 to get other movie
bckmv=bckmv(1,:);
%
% this part is directly copy from motion code
%%
b = bckmv(1:28);
c = bckmv(29:56);
k = bckmv(57:84);
m = bckmv(85:92)';
v1= bckmv(93:100)';
v2= bckmv(101:108)';
v3= bckmv(109:116)';
%% Define global variables:
global g
 g = -9.81; % in m/s^2 %%%%%%
global dt
 dt = .001;%%%%%%%%%%
global T;
 T = 0;

kg = 50 ;%% %%%%%%%%%%%
u = 0.5;%%
damp=0.29;
L0 = .1;
Vo = [v1,v2,v3];
V = [Vo(:,1),Vo(:,2),Vo(:,3)];
[l]=lengthorigin(Vo);
[idx_k0] = find(k==0); % find the k =0 which means the spring is not exist
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
while n<= 10/dt % change motion time to 10 s
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




 T = T+dt;
end
tdistance = sqrt((mean(posx))^2 + (mean(posy))^2);
%% Plot part
ccc=0;
E=[ 1 4;2 3;5 8;6 7; % link of spring
 1 2;3 4;5 6;7 8;
 1 5;2 6;3 7;4 8;
 1 3;2 4;5 7;6 8;
 1 8;4 5;2 7;3 6;
 1 6;2 5;3 8;4 7;
 1 7;3 5;2 8;4 6];
E(idx_k0,:) = []; % delete springs which not exist
for loop=10:10:10000 % create movie
 V=[Distancex(:,loop) Distancey(:,loop) Distancez(:,loop)];
 figure(1)
 plot3(V(:,1),V(:,2),V(:,3),'.','Markersize',20,'color',[0.8500 0.3250 0.0980]);
hold on;
for i=1:size(E,1)
V1=V(E(i,1),:);
V2=V(E(i,2),:);
line([V1(1) V2(1)],[V1(2) V2(2)],[V1(3) V2(3)],'LineWidth',1.2);
end
axis equal
axis([-1 15 -1 15 -1 3]) % set range, could adjust to make video more clearly
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
movie(M,1,100) % show movie in 100 fps
