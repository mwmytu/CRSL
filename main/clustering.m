function [NMI,ACC,f,RI,AR,p]=clustering(S, cls_num, gt)

[C] = SpectralClustering(S,cls_num);
[~, NMI, ~] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));
[f,~,~] = compute_f(gt,C);
[AR,RI,~,~]=RandIndex(gt,C);

if size(C,2) ~= 1
    C = C';
end
if size(gt,2) ~= 1
    gt = gt';
end

n = length(C);

uY = unique(C);
nclass = length(uY);
Y0 = zeros(n,1);
if nclass ~= max(C)
    for i = 1:nclass
        Y0(find(C == uY(i))) = i;
    end
    C = Y0;
end
uY = unique(gt);
nclass = length(uY);
predY0 = zeros(n,1);
if nclass ~= max(gt)
    for i = 1:nclass
        predY0(find(gt == uY(i))) = i;
    end
    gt = predY0;
end
Lidx = unique(C); classnum = length(Lidx);
predLidx = unique(gt); pred_classnum = length(predLidx);
% purity
correnum = 0;
for ci = 1:cls_num
    incluster = C(find(gt == predLidx(ci)));
    inclunub = histcounts(incluster, 1:max(incluster)); 
    if isempty(inclunub) 
        inclunub=0;
    end
    correnum = correnum + max(inclunub);
end
p = correnum/length(gt);
end
