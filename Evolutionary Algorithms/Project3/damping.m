clear all
close all
clc
%% Initialization (All the components of this code are the same as bouncing part, except for adding dampening):
T = 3;
dt = 1e-3;
vz=zeros(1,T/dt);
vdz=zeros(1,T/dt);
vx=zeros(1,T/dt);
vy=zeros(1,T/dt);
h=zeros(1,T/dt);
s=zeros(1,T/dt);
dx=zeros(1,T/dt);
dy=zeros(1,T/dt);
dz=zeros(1,T/dt);
g=-9.81;
t=0;
l=0.1;
h(1)=.8;
s(1)=.8;
k=500;
m=0.1;
az=g;
ax=0;
ay=0;
adz=0;
damp = .9; % This is dampening coefficient
%% Reaction calculation:
for i =2 : (T/dt)
 s(i)=s(i-1)+vz(i-1)*dt;
 h(i)=s(i);
 vz(i)=az*dt+vz(i-1);
 dx(i)=dx(i-1)+2*vx(i-1)*dt;
 dy(i)=dy(i-1)+2*vy(i-1)*dt;
 dz(i)=dz(i-1)+2*vdz(i-1)*dt;

 ly=l-dy(i);
 lx=l-dx(i);
 lz=l-dz(i);
 fx=-k*dx(i);
 fy=-k*dy(i);
 fz=-k*dz(i);
 
 fxy=-k*(l*2^0.5-(ly^2+lx^2)^0.5);
 fxz=-k*(l*2^0.5-(lz^2+lx^2)^0.5);
 fyz=-k*(l*2^0.5-(ly^2+lz^2)^0.5);
 fxyz=-k*(l*3^0.5-(ly^2+lz^2+lx^2)^0.5);

 ax=fx/m*4+fxy*lx/(lx^2+ly^2)^0.5/m*4+fxz*lx/(lx^2+lz^2)^0.5/m*4+fxyz*lx/(lx^2+ly^2+lz^2)^0.5/m*4;
 ay=fy/m*4+fxy*ly/(lx^2+ly^2)^0.5/m*4+fyz*ly/(ly^2+lz^2)^0.5/m*4+fxyz*ly/(lx^2+ly^2+lz^2)^0.5/m*4;

adz=fz/m*4+fxz*lz/(lz^2+lx^2)^0.5/m*4+fyz*lz/(lz^2+ly^2)^0.5/m*4+fxyz*lz/(lx^2+ly^2+lz^2)^0.5/m*4;
 vx(i)=ax*dt+vx(i-1);
 vy(i)=ay*dt+vy(i-1);
 vdz(i)=adz*dt+vdz(i-1);

 if s(i)<=0
 h(i)=0;
 dz(i)=dz(i-1)-s(i);
 s(i)=0;
 vz(i)= (((0.5*vz(i-1)^2+(-g*s(i-1))/2)*2-(g*s(i)/2))^0.5)*damp;
 else
 az=g ;
 end

 Ex=.5*k*dx(i)^2;
 Ey=.5*k*dy(i)^2;
 Ez=.5*k*dz(i)^2;
 Exy=.5*k*(l*2^0.5-(ly^2+lx^2)^0.5)^2;
 Exz=.5*k*(l*2^0.5-(lz^2+lx^2)^0.5)^2;
 Eyz=.5*k*(l*2^0.5-(ly^2+lz^2)^0.5)^2;
 Exyz=.5*k*(l*3^0.5-(ly^2+lz^2+lx^2)^0.5)^2;
 Espring(i) = (Ex+Ey+Ez+Exy+Exz+Eyz+Exyz)*4;

 t=t+dt;

end
%% Visualization:
E=[ 1 2;1 3;1 4;1 5;
 1 6;1 7;1 8;2 3;
 2 4;2 5;2 6;2 7;
 2 8;3 4;3 5;3 6;
 3 7;3 8;4 5;4 6;
 4 7;4 8;5 6;5 7;
 5 8;6 7;6 8;7 8];
n = 0;
for loop=1: T/dt/300: T/dt
 V=[-dx(loop)/2 -dy(loop)/2 h(loop);
 -dx(loop)/2 l+dy(loop)/2 h(loop);
 l+dx(loop)/2 l+dy(loop)/2 h(loop);
 l+dx(loop)/2 -dy(loop)/2 h(loop);
 -dx(loop)/2 -dy(loop)/2 h(loop)+l+dz(loop);
 -dx(loop)/2 l+dy(loop)/2 h(loop)+l+dz(loop);
 l+dx(loop)/2 l+dy(loop)/2 h(loop)+l+dz(loop);
 l+dx(loop)/2 -dy(loop)/2 h(loop)+l+dz(loop);];

figure(1)

plot3(V(:,1),V(:,2),V(:,3),'.','Markersize',18,'color',[0.8500 0.3250 0.0980]);
hold on;
for i=1:size(E,1)
V1=V(E(i,1),:);
V2=V(E(i,2),:);
line([V1(1) V2(1)],[V1(2) V2(2)],[V1(3) V2(3)],'LineWidth',1.2);
end
axis equal
axis([-0.2 0.2 -0.2 0.2 0 .95])
grid minor
ylabel('Y (m)')
xlabel('X (m)')
zlabel('Z (m)')
%title('Bouncing Cube w/ Dampening Coefficient = 0.9')
hold off
n = n+1;
 M(n)=getframe(gcf);
end
vtotal = sqrt(vx.^2 + vy.^2 + vz.^2 + vdz.^2);
KE = .5*8*m.*vtotal.^2;
PE = -(8*m*g.*s + 4*m*g.*(l-dz));
E = KE + PE + Espring;
figure(2)
plot(linspace(0,3,length(h)),KE);
hold on
grid minor
plot(linspace(0,3,length(h)),PE);
plot(linspace(0,3,length(h)),E);
legend('KE','PE','E');
%title('Energy Plot w/ Dampening Coefficient = 0.9');
ylabel('Energy (J)');
xlabel('Time (s)');
%% Generate video:
video = VideoWriter('Bounce_Damp.avi');
open(video);
writeVideo(video,M);
close(video);

