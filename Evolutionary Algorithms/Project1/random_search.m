clear all
close all

data=load('tsp.txt');
S=size(data,1);
N=1000;
numIter=1e5;

%% select the poins randomly
SampleRows = randperm(S); 
SampleRows = SampleRows(1:N); 
Sample_data = data(SampleRows,:);


a = meshgrid(1:N);
dmat = reshape(sqrt(sum((Sample_data(a,:)-Sample_data(a',:)).^2,2)),N,N); %calculate the distance
plot(Sample_data(:,1),Sample_data(:,2),'k.')

V=zeros(numIter,N+1);
total_distance=zeros(numIter,1);
best_solution=zeros(numIter,1);
globalMax=0;
for it=1:numIter
    x=randperm(N); 
    V(it,:)=[x,x(1,1)];  
for i=1:N              %calculate the distance between two cities for every closed path
    d(i)=dmat(V(it,i),V(it,1+i));
end
total_distance(it,1)=sum(d);
if total_distance(it,1) > globalMax
        globalMax= total_distance(it,1);
        bestRoute = V(it,:);
        plot(Sample_data(bestRoute,1),Sample_data(bestRoute,2),'r.-');
        %title(sprintf('Total Distance = %1.4f, Iteration = %d',total_distance(it,1),it));
pause(0.01)
end
 best_solution(it,1)=globalMax;
end
figure('Name','Random Search','Numbertitle','off');
scatter(1:numIter,best_solution,'b.');


    





