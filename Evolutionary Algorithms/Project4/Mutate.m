%ode for Mutation
function output = Mutate(a)
 % choose point for b c k
 randnum1 = randperm(length(a)/3,1);
 randnum2 = randperm(length(a)/3,1)+28;
 randnum3 = randperm(length(a)/3,1)+56;
 %mutate
 a(randnum1(1)) = rand*(.02)-0.01;
 a(randnum2(1)) = rand(1);
 a(randnum3(1)) = rand*(500)+500;
 output = a;
end