%% Initialization:
% Start iterating time
clearvars
close all
clc
n = 28;
w = pi*10;
%% Genetic Algorithm
run = 3; % running times
iter = 2500; % evaluate times
popnum = 50; % number of population
path_new = NaN(iter, n+1);
kept = NaN(iter, 1);
tdistance = NaN(popnum, 1);
Tdistance = NaN(popnum,iter);
cood=6;
alldata = NaN(iter,run);
bckmv=zeros(popnum,(28*3+8+24));
for j = 1:run
    % Generate initial populations randomly:
pop = NaN(popnum, n);
for i=1:popnum
 springnum=randi([0, 1], [28, 1]);% decide number and positon of springs
 b = springnum .*(rand(28,1)*(0.5)-0.25); % range of b
 c = springnum .*rand(28,1)*2*pi; %range of c
 k =springnum .*( rand(28,1)*(20)+30);% range of k

 mm=rand(8,1);% decide mass distribution
 m=mm*0.8/sum(mm);
 vn=randperm(cood^3,8);
 v=locationsix(vn,cood)';

 bckmv(i,:) = [b;c;k;m;v]; % create gene
end
for N = 1:iter
 tic
 for i = 1:popnum
 tdistance(i) = Motion(w,bckmv(i,:));
 end
 tdistance(isnan(tdistance))=0;
 Tdistance(:,N) = tdistance; % Store all the distances
 if N == 800 || N == 1200 || N == 2500
 filename = strcat('data_', num2str(j), '_', num2str(N), '.mat');
 save(filename);
 end
 [path_best,idx_best] = max(tdistance); % Find the longest path and its position in matrix
 fit = tdistance; % value of fitness

 % store the best value:
 kept(N) = path_best;
 bckmv2=zeros(15,(28*3+8+24));
 bckmv3=zeros(15,(28*3+8+24));


 for i = 1:15
 bckmv1 = Rank(bckmv,fit); % Tournament Selection
 bckmv2(i,:) = Cross(bckmv1); % Crossover

 nm= randperm(popnum,1);
 bckmv3(i,:) = Mutate(bckmv(nm,:)); % Mutation

 end

 [~,idx] = sort(tdistance,'descend');

 bckmv = [bckmv(idx(1:20),:);bckmv2;bckmv3]; % Get good gene from parents


toc
end
alldata(:,j) = kept;
end
