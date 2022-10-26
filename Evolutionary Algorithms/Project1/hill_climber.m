clear all
close all
xy =load('tsp.txt');
N = size(xy,1);
a = meshgrid(1:N);
dmat = reshape(sqrt(sum((xy(a,:)-xy(a',:)).^2,2)),N,N);
popSize = 50;
numIter = 1e5;
showProg = 1;
showResult = 1;

% Verify Inputs
[N,dims] = size(xy);
[nr,nc] = size(dmat);
if N ~= nr || N ~= nc
    error('Invalid XY or DMAT inputs!')
end
n = N;

% Sanity Checks
popSize = 4*ceil(popSize/4);
numIter = max(1,round(real(numIter(1))));
showProg = logical(showProg(1));
showResult = logical(showResult(1));

% Initialize the Population
pop = zeros(popSize,n);
pop(1,:) = (1:n);
for k = 2:popSize
    pop(k,:) = randperm(n);
end

% Run the GA
globalMin = Inf;
totalDist = zeros(popSize,1);
distHistory = zeros(1,numIter);
tmpPop = zeros(popSize,n);
newPop = zeros(popSize,n);
if showProg
    pfig = figure('Name','Hill Climb','Numbertitle','off');
end
for iter = 1:numIter
    % Evaluate Each Population Member (Calculate Total Distance)
    for p = 1:popSize
        d = dmat(pop(p,n),pop(p,1)); % Closed Path
        for k = 2:n
            d = d + dmat(pop(p,k-1),pop(p,k));
        end
        totalDist(p) = d;
    end

    % Find the Best Route in the Population
    [minDist,index] = min(totalDist);
    distHistory(iter) = minDist;
    if minDist < globalMin
        globalMin = minDist;
        optRoute = pop(index,:);
        if showProg
            % Plot the Best Route
            figure(pfig);
            rte = optRoute([1:n 1]);
            if dims > 2, plot3(xy(rte,1),xy(rte,2),xy(rte,3),'r.-');
            else plot(xy(rte,1),xy(rte,2),'r.-'); end
            %title(sprintf('Total Distance = %1.4f, Iteration = %d',minDist,iter));
        end
    end

    % Genetic Algorithm Operators
        dists = totalDist;
        [Ordered_dis, index]=sort(dists,'ascend');
        half=index(1:size(index)/2,1);
        Top_half=pop(half,:);
        routeInsertionPoints = sort(ceil(n*rand(1,2)));
        I = routeInsertionPoints(1);
        J = routeInsertionPoints(2);
        tmpPop=Top_half;
        tm = tmpPop(:,I);
        tmpPop(:,I) = tmpPop(:,J);
        tmpPop(:,J) = tm;
        newPop=[tmpPop;Top_half];
        pop = newPop;
end
%% plot result
if showResult
    % Plots the GA Results
    figure('Name','TSP_GA | Results','Numbertitle','off');
    subplot(2,2,1);
    pclr = ~get(0,'DefaultAxesColor');
    if dims > 2, plot3(xy(:,1),xy(:,2),xy(:,3),'.','Color',pclr);
    else plot(xy(:,1),xy(:,2),'.','Color',pclr); end
   % title('City Locations');
    subplot(2,2,2);
    rte = optRoute([1:n 1]);
    if dims > 2, plot3(xy(rte,1),xy(rte,2),xy(rte,3),'r.-');
    else plot(xy(rte,1),xy(rte,2),'r.-'); end
    %title(sprintf('Total Distance = %1.4f',minDist));
    subplot(2,2,3);
    plot(distHistory,'b','LineWidth',2);
    %title('Best Solution History');
    set(gca,'XLim',[0 numIter+1],'YLim',[0 1.1*max([1 distHistory])]);
end

