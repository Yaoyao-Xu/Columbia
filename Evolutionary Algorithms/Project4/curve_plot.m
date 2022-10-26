clear  all
clc
load( 'data.mat')  %% Learning Curve  
x=2500;  avel= mean(alldata,2); % calculated the mean  
errorl=std(alldata,0,2)/4^0.5; % plot learning curve  
X=[1:2500];  
plot(X,avel)  
hold on  
errorbar(((2500/10):(2500/10):2500),avel((2500/10):(2500/10):2500),errorl((2500/10):(2500/10):2500),'o')  
legend('GA','Error Bar')  
%title('Learning Curve','FontSize',14)  
ylabel('Distance(m)','FontSize',14)  
xlabel('Number of Evaluation','FontSize',14)   
%% Dot Plot    
xx=[];  Dot=[];  
Tdistance=Tdistance';  
for i= 1:20 %% set x for dot plot  
    xx=[xx 1:2500];  
    Dot=[Dot;Tdistance(:,i)];  
end
figure(2)  
scatter(xx,Dot,2,'g') %dot plot  
hold on  
plot(X,alldata(:,3))%best fitness  
legend('Dot','Best Fitness')  
%title('Dot plot of the Best 40% ','FontSize',14)  
ylabel('Distance(m)','FontSize',14)  
xlabel('Number of Evaluation','FontSize',14)  
hold off   %% Convergence  %Set compare point  
for i =1:2500  
    g1=Tdistance(i,:);  
    number1=length(g1(g1>=2));  
    prop1(i)=number1/50;  
end
for i =1:2500  
    g2=Tdistance(i,:);  
    number2=length(g2(g2>=5));  
    prop2(i)=number2/50;  
end
for i =1:2500  
    g3=Tdistance(i,:);  
    number3=length(g3(g3>=8)); 
    prop3(i)=number3/50;  
end  % Plot  
figure(3)  
plot([1:2500],prop1,[1:2500],prop2,[1:2500],prop3)  
legend('Distance>2','Distance>5','Distance>8')  
%title('Convergence from GA','FontSize',14)  
ylabel('Proportion','FontSize',14)  
xlabel('Evaluation','FontSize',14)