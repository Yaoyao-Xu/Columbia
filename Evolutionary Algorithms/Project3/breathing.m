clear all
close all
clc

%% Initialization:
% Define global variables:
global g
g = [0,0,-9.81]; % in m/s^2
global dt
dt = .01;
global T;
T = 0;
A = .0005; % amplitude of breathing
w = 10; % frequency of breathing
k=500;
L0 = .1; % original length
Vo = [0 0 0; 0 L0 0; L0 L0 0; L0 0 0; 0 0 L0; 0 L0 L0; L0 L0 L0; L0 0 L0]; % vertices matrix
V = [Vo(:,1)+.1,Vo(:,2)+.1,Vo(:,3)+.1]; % move the position in advantage of good vision
posx = V(:,1);
posy = V(:,2);
posz = V(:,3);
n = 0; % initialize the frame count
while (1)
 for j = 1:length(V)
 if j == 1 || j == 2 || j == 5 || j == 6 % left side of the cube
 posx(j) = posx(j) - A*sin(w*T);
 else % right side of the cube
 posx(j) = posx(j) + A*sin(w*T);
 end
 if j == 1 || j == 4 || j == 5 || j == 8 % front side of the cube
 posy(j) = posy(j) - A*sin(w*T);
 else % back side of the cube
 posy(j) = posy(j) + A*sin(w*T);
 end
 if j == 5 || j == 6 || j == 7 || j == 8 % top side of the cube
 posz(j) = posz(j) + A*sin(w*T);
 else % bottom side of the cube
 posz(j) = posz(j) - A*sin(w*T);
 end
end
n = n+1;
dl(n) = abs(posx(1,1)-posx(4,1))-L0; % calculate the displacement between two vertices
 V = [posx,posy,posz];

 E = [1 2; 2 3; 3 4; 4 1; 5 6; 6 7; 7 8; 8 5; 1 5; 2 6; 3 7; 4 8;...
 1 8; 4 5;...
 2 5; 1 6;...
 3 8; 4 7;...
 2 7; 3 6;...
 1 3; 2 4;...
 5 7; 6 8;...
 1 7; 2 8;...
 3 5; 4 6];
%% Visualization:
 figure(1)
 plot3(V(:,1),V(:,2),V(:,3),'.','Markersize',20,'color',[0.8500 0.3250 0.0980]);
 %title('Breathing Cube');
 xlabel('X (m)');
 ylabel('Y (m)');
 zlabel('Z (m)');
 grid on
 grid minor
 hold on
 for i = 1:size(E,1)
 V1 = V(E(i,1),:);
 V2 = V(E(i,2),:);
 line([V1(1) V2(1)],[V1(2) V2(2)],[V1(3) V2(3)],'LineWidth',1.5);
 end
 axis equal
 axis([0 L0+.2 0 L0+.2 0 L0+.2])
 set(gca,'XDir','reverse')
 set(gca,'YDir','reverse')
 set(gcf,'doublebuffer','on');
 T = T+dt;
 hold off
 M(n) = getframe(gcf);
 if n >= 100
 break;
 end
end
%% Plot Energy:
E = 12*.5*k.*dl.^2 + 12*.5*k.*(sqrt(2)*dl).^2 + 4*.5*k.*(sqrt(3)*dl).^2; 
figure(2)
plot(linspace(0,3,length(dl)),E);
grid minor
%title('Energy Plot of Breathing Cube');
xlabel('Time (s)');
ylabel('Energy (J)');
%% Generate video:
video = VideoWriter('Breath.avi');
open(video);
writeVideo(video,M);
close(video);
