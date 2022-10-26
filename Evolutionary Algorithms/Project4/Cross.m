function output = Cross(chrom1)
 a = chrom1(1,:);%parent1
 b = chrom1(2,:);%parent2
 %choosee point for b c k
 randnum1 = randperm(length(a)/3,2);
 randnum2 = randperm(length(a)/3,2)+28;
 randnum3 = randperm(length(a)/3,2)+56;
 %cross the point to create child
 a(randnum1(1)) = b(randnum1(2));
 a(randnum2(1)) = b(randnum2(2));
 a(randnum3(1)) = b(randnum3(2));
 output = a;

end
