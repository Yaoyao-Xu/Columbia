function chrom_selected = select(chrom,fit)
n = length(fit);
for i=1:1:n % devide roulette wheel into n parts
 adapt(i)=(fit(i))/(sum(fit));
end
% choose first parent
i=2;
select=rand;
value=adapt(1);
while value<select
 value=value+adapt(i);
 i=i+1;
end
s1=i-1;
chrom_selected(1,:) = chrom((s1),:);
s2=s1;
% choose second parent
while s2==s1
 i=2;
 select=rand;
 value=adapt(1);
 while value<select
 value=value+adapt(i);
 i=i+1;
 end
 s2=i-1;

end
chrom_selected(2,:) = chrom(s2,:);
end

 