function [v] = locationsix(vn,cood)
% find the start position of masses
v=zeros(1,length(vn)*3);
% ensure the coordinate
p=zeros(cood^3,3);
n=1;
for a =1:cood
 for b=1:cood
 for c=1:cood
 p(n,:)=[(a-1), (b-1), (c-1)];
 n=n+1;
 end
 end
end
% find the masses position
p=p*0.5;
for i= 1: length(vn)
v(i)=p(vn(i),1);
v(i+length(vn))=p(vn(i),2);
v(i+length(vn)*2)=p(vn(i),3);
end
