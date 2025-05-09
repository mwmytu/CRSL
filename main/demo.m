load("ORL.mat"); 
% load('NGs.mat');
% load('MSRC_v1.mat');
% load('BBCSport.mat');
% load('BBC.mat');
% load('100leaves.mat')
% load('COIL20.mat')
% load('Handwritten10_6_2k.mat');
fprintf('Begin\n');

% 初始化参数
% X = data;
% X{1} = X{1}';X{2} = X{2}';X{3} = X{3}';
% gt=label';
V = size(X, 2);  
usePCA = true;  
opts = [];  
opts.ReducedDim = 100;  
numSamples = length(gt);  
numClust = size(unique(gt), 1);  
paramGrid = struct('lambda1',0.001, 'lambda2', 0.1, 'lambda3', 10, 'lambda4', 0.01,'lambda5',0.001,'lambda6',10,'lambda7',10);% orl
% paramGrid = struct('lambda1',0.001, 'lambda2', 0.01, 'lambda3', 0.01,'lambda4', 0.001,'lambda5',0.001,'lambda6',0.1 ,'lambda7',1);% ngs
% paramGrid = struct('lambda1',0.01, 'lambda2', 0.1, 'lambda3', 1,'lambda4', 0.01,'lambda5', 1,'lambda6',0.01 ,'lambda7',1);%MSRC_V1
% paramGrid = struct('lambda1',0.001, 'lambda2', 0.01, 'lambda3', 0.01, 'lambda4', 0.01,'lambda5',0.01,'lambda6',0.1,'lambda7',1 );% bbcs
% paramGrid = struct('lambda1',0.1, 'lambda2', 0.01, 'lambda3', 0.001, 'lambda4',0.1,'lambda5',0.001,'lambda6',10,'lambda7',100);% bbc
% paramGrid = struct('lambda1',0.01, 'lambda2', 0.01, 'lambda3', 0.001, 'lambda4', 0.01,'lambda5',0.0001,'lambda6',0.1,'lambda7',1);% 100leaves
% paramGrid = struct('lambda1',0.0001, 'lambda2', 0.001, 'lambda3', 1000, 'lambda4', 0.001,'lambda5',0.0001,'lambda6',1,'lambda7',1  );% coil20
% paramGrid = struct('lambda1',0.0001, 'lambda2', 0.01, 'lambda3', 0.01, 'lambda4', 0.01,'lambda5',0.01,'lambda6',0.01 ,'lambda7',0.1);% hw
bestResult = struct('meanNMI', 0, 'stdNMI', 0, 'meanACC', 0, 'stdACC', 0, 'meanF', 0, 'stdF', 0, 'meanRI', 0, 'stdRI', 0, 'meanAR', 0, 'stdAR', 0, 'meanP', 0, 'stdP', 0);  

for i = 1:V
    if size(X{i}, 1) ~= numSamples
        X{i} = X{i}';
    end
    if usePCA && size(X{i}, 2) > opts.ReducedDim
        [P1, ~] = PCA1(X{i}, opts);  
        X{i} = X{i} * P1;
    end
    X{i} = X{i}';  
end
for i = 1:V
    X{i}(X{i} < 0) = 0; 
    X{i} = abs(X{i}); 
end

for i = 1:length(paramGrid.lambda1)
    for j = 1:length(paramGrid.lambda2)
        for k = 1:length(paramGrid.lambda3)
            for l = 1:length(paramGrid.lambda4)
                for a = 1:length(paramGrid.lambda5)
                    for b = 1:length(paramGrid.lambda6)
                        for c = 1:length(paramGrid.lambda7)
                            paras.m = 100;
                            paras.mu = 1e-4;
                            paras.lambda1 = paramGrid.lambda1(i);
                            paras.lambda2 = paramGrid.lambda2(j);
                            paras.lambda3 = paramGrid.lambda3(k);
                            paras.lambda4 = paramGrid.lambda4(l);
                            paras.lambda5 = paramGrid.lambda5(a);
                            paras.lambda6 = paramGrid.lambda6(b);
                            paras.lambda7 = paramGrid.lambda7(c);
                         
                            results = zeros(2, 6);

                            for trial = 1:2
                                [NMI, ACC, F, RI, AR, P] = CRSL(X, gt, numClust, paras);
                                results(trial, :) = [NMI, ACC, F, RI, AR, P];
                            end
                            sortedResults = sort(results);

                            meanResult = mean(sortedResults);
                            stdResult = std(sortedResults);

                            if meanResult(2) > bestResult.meanACC
                                bestResult.meanNMI = meanResult(1);
                                bestResult.stdNMI = stdResult(1);
                                bestResult.meanACC = meanResult(2);
                                bestResult.stdACC = stdResult(2);
                                bestResult.meanF = meanResult(3);
                                bestResult.stdF = stdResult(3);
                                bestResult.meanRI = meanResult(4);
                                bestResult.stdRI = stdResult(4);
                                bestResult.meanAR = meanResult(5);
                                bestResult.stdAR = stdResult(5);
                                bestResult.meanP = meanResult(6);
                                bestResult.stdP = stdResult(6);
                                bestResult.params = paras;
                            end
                        end
                    end
                end
            end
        end
    end
end
fprintf('最佳结果:\nNMI: %f ± %f\nACC: %f ± %f\nF: %f ± %f\nRI: %f ± %f\nAR: %f ± %f\nP: %f ± %f\n', bestResult.meanNMI, bestResult.stdNMI, bestResult.meanACC, bestResult.stdACC, bestResult.meanF, bestResult.stdF, bestResult.meanRI, bestResult.stdRI, bestResult.meanAR, bestResult.stdAR, bestResult.meanP, bestResult.stdP);
fprintf('最佳参数:\nlambda1: %f\nlambda2: %f\nlambda3: %f\nlambda4: %f\nlambda5: %f\nlambda6: %f\nlambda7: %f\n', bestResult.params.lambda1, bestResult.params.lambda2, bestResult.params.lambda3, bestResult.params.lambda4,bestResult.params.lambda5,bestResult.params.lambda6,bestResult.params.lambda7);
