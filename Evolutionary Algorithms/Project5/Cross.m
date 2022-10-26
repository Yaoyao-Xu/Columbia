function output = Cross(chrom1)
 a = chrom1(1,:); % set parents
 b = chrom1(2,:);
 randnum1 = randi((length(a)-32)/3,[2,1]); % choose exchange point for b c k m position
 randnum2 = randi((length(a)-32)/3,[2,1])+28;
 randnum3 = randi((length(a)-32)/3,[2,1])+56;
 mass=randi([2],1);
 coordinate=randi([2],1);

 a(randnum1(1)) = b(randnum1(2)); % create child
 a(randnum2(1)) = b(randnum2(2));
 a(randnum3(1)) = b(randnum3(2));
 a(85:1:92)=chrom1(mass,85:1:92);
 a(93:1:116)=chrom1(coordinate,93:1:116);
 output = a;

end
