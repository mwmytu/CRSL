function [NMI,ACC,f,RI,AR,p] = CRSL(X,gt,cls_num,paras)
V = size(X,2);
N = size(X{1},2); 
knn = 5;
lambda1= paras.lambda1;
lambda2= paras.lambda2;
lambda3= paras.lambda3;
lambda4= paras.lambda4;
lambda5= paras.lambda5;
lambda6= paras.lambda6;
lambda7= paras.lambda7;
mu = paras.mu;
m = paras.m;
%% Normalize X
for i=1:V
    X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);
end
%% Initialize variables
for i = 1:V
    [d,~]=size(X{i});
    P{i} = zeros(d,m);
    D{i} = zeros(m,N);  
    Z{i} = zeros(N,N);
    W{i} = zeros(N,N);
    E{i} = zeros(N,N);
    J{i} = zeros(N,N);
    Y1{i} = zeros(N,N);
    Y2{i} = zeros(N,N);
end
K = V;
H = rand(m,N);
S = zeros(N,N);
y1 = zeros(N*N*K,1); g = zeros(N*N*K,1);
dim1 = N;dim2 = N;dim3 = K;
sX = [N, N, K];
I   = eye(N);
[L,~] = constructG(X, knn, V, N);

sumL = sparse(size(L{1}, 1), size(L{1}, 2)); 
for v = 1:V
    sumL = sumL + L{v};
end
%% Algorithm parameters
epsilon = 1e-7;
pho = 2;max_mu = 1e5;
max_iter = 100;
iter = 1;
converge_Z=[];
IsConverge = 0;
%% Main loop
iter_time = zeros(1,max_iter);  
total_time = 0;                
tic;  
while (IsConverge == 0&&iter<max_iter+1)
    fprintf('---------processing iter %d--------\n', iter+1);
    iter_tic = tic;
    %% update P
    for i = 1:V
        P{i} = updateP(X{i},H+D{i}); 
    end

    %% update H                  
    for i = 1:V
        A = 2*lambda1 * P{i}' * P{i};
        B = 2 * lambda3 * (I-S)*(I-S');
        C = 2 *lambda1* P{i}'*P{i}* D{i}-2*lambda1*P{i}'*X{i};
    end    
    A = A + epsilon * eye(size(A));
    B = B + epsilon * eye(size(B));
    H = lyap(A, B, C);
    clear A; clear B; clear C;

    %% update D 
    M = eye(N) - (1 / N) * ones(N); 
    for i = 1:V
    A = 2*lambda1*(P{i}')*P{i};  
    B = 2*lambda5*M*D{i}'*D{i}*M;
    A = A + epsilon * eye(size(A));
    B = B + epsilon * eye(size(B));
    C = 2 *lambda1* P{i}'*P{i}* H-2*lambda1*P{i}'*X{i};
    D{i} = lyap(A,B,C);
    end 

    %% update W
    for i = 1:V
        W{i} = ((2*lambda4*I+mu*I)\eye(N))*(2*lambda4*S+mu*(Z{i}-E{i}+Y2{i}/mu)); 
    end
    %% update E 
    for i = 1:V
        E{i} = ((2*lambda7*I+mu*I)\eye(N))*(mu*(Z{i}-W{i}+Y2{i}/mu));
    end
    %% update S
    for i = 1:V
        B = 2 * lambda6 * I+2*lambda6*sumL;
        C = -2 *lambda3* (H')*H-2*lambda4*W{i};
    end  
    A = 2*lambda3 * (H') * H;
    A = A + epsilon * eye(size(A));
    B = B + epsilon * eye(size(B));
    S = lyap(A, B, C);
    clear A; clear B; clear C;
    %% update Z
    for i = 1:V
        Z{i} = ((2*lambda2*(X{i}')*X{i}+2*mu*I)\eye(N)) * (2*lambda2*X{i}' * X{i}+mu*(J{i}+W{i}+E{i}-Y1{i}/mu-Y2{i}/mu));
    end
    %% update J
    Z1_tensor = cat(3, Z{:,:});
    Y1_tensor = cat(3, Y1{:,:});
    z = Z1_tensor(:);
    y1 = Y1_tensor(:);
    [g, ~] = wshrinkObj_weight(z + 1/mu*y1,1/mu,sX,0,3);
    J_tensor = reshape(g, sX);

   for i = 1:V
        J{i} = J_tensor(:,:,i);
        Y1{i} = Y1{i}+mu*(Z{i}-J{i});
        Y2{i} = Y2{i}+mu*(Z{i}-W{i}-E{i});
   end
 
    %% updata multipliers
    mu = min(pho*mu, max_mu);
    %% convergence conditions
    thrsh = 1e-7;
    max_Z=0;
    for i =1:V
    if(norm(Z{i}-J{i},"fro")>=thrsh && norm(Z{i}-W{i}-E{i},"fro")>=thrsh)    

        IsConverge = 0;
    else
        IsConverge = 1;
    end
    end
    iter_time(iter) = toc(iter_tic);      
    iter = iter + 1;
end

disp('======= 时间统计 =======');
disp(['单次迭代耗时: ',num2str(mean(iter_time(1:iter-1))),'±',num2str(std(iter_time(1:iter-1))),' 秒']);
disp(['总运行时间: ',num2str(toc),' 秒']); 

Z_all = 0;
for i = 1:V
    Z_all = Z_all+abs(E{i})+abs(E{i}');
end
Z_all = Z_all/V;
[NMI,ACC,f,RI,AR,p]=clustering((abs(S)+abs(S')+Z_all)/2, cls_num, gt);
