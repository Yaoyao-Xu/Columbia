function chrom_selected = Rank(chrom,fit)
%% tournament selection
num=length (fit);
choose=randperm(num,7); % choose 7 group member
score=fit(choose); % rank them
name=chrom(choose,:);
[~,idx] = sort(score,'descend'); % choose best 2 of them
chrom_selected=name(idx(1:2),:);
end
