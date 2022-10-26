function output = Mutate(a)
 randnum1 = randperm((length(a)-32)/3,2); % choose mutate point
 randnum2 = randperm((length(a)-32)/3,2)+28;
 randnum3 = randperm((length(a)-32)/3,2)+56;
 randnum4 = randperm(8,2)+84;
 randnum5 = randperm(8,2)+92;
 randnum6 =randnum5+8;
 randnum7 =randnum5+16;
 gene=[a(randnum1(1)),a(randnum2(1)),a(randnum3(1))];
 a(randnum1(1)) = a(randnum1(2)); % mutation b c k
 a(randnum2(1)) = a(randnum2(2));
 a(randnum3(1)) = a(randnum3(2));
 a(randnum1(2)) = gene(1);
 a(randnum2(2)) = gene(2);
 a(randnum3(2)) = gene(3);
 % mutation mass distribution
 total=a(randnum4(1))+ a(randnum4(2)) ;
 mm=rand*( total);
 a(randnum4(1))=mm;
 a(randnum4(2))=total-mm;

 % mutation mass position
 cha=[a(randnum5(1)),a(randnum6(1)) ,a(randnum7(1)) ];
 a(randnum5(1))=a(randnum5(2));
 a(randnum6(1))=a(randnum6(2));
 a(randnum7(1))=a(randnum7(2));

 a(randnum5(2))=cha(1);
 a(randnum6(2))=cha(2);
 a(randnum7(2))=cha(3);
 output = a;
end
