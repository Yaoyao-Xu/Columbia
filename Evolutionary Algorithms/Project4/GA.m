%% Initialization:
tic % Start itering time
clearvars
close all
clc
n = 28;
w = 3*pi;
%% Genetic Algorithm
run = 3; % runtimes
iter = 2500;%evolve times
popnum = 50; % number of population
path_new = NaN(iter, n+1);
kept = NaN(iter, 1);
tdistance = NaN(popnum, 1);
Tdistance = NaN(popnum,iter);
alldata = NaN(iter,run);
bck=zeros(popnum,28*3);
for j = 1:run
% Generate initial populations randomly:
pop = NaN(popnum, n);
for i=1:popnum
 %create b c k
 b = rand(28,1)*(.02)-0.01; % -0.5 ~ 0.5
 c = rand(28,1); %
 k = rand(28,1)*(1000-500)+500;
 bck(i,:) = [b;c;k];
end
for N = 1:iter
 %tic
 for i = 1:popnum
 tdistance(i) = Motion(w,bck(i,:));
 end
 Tdistance(:,N) = tdistance; % Store all the distances
 [path_best,idx_best] = max(tdistance); % Find the longest path and its position in
 fit = tdistance; % value of fitness

 % store the best value:
 kept(N) = path_best;
 bck2=zeros(15,28*3);
 bck3=zeros(15,28*3);


 for i = 1:15
 bck1 = select(bck,fit); % Roulette Wheel Selection
 bck2(i,:) = Cross(bck1); % Crossover

 nm= randperm(popnum,1);
 bck3(i,:) = Mutate(bck(nm,:)); % Mutation

 end




 [~,idx] = sort(tdistance,'descend');

 bck = [bck(idx(1:20),:);bck2;bck3]; % Get good gene from parents
 %toc
 disp(N)
end
alldata(:,j) = kept;%record the distance
save ('data.mat')
end
toc
