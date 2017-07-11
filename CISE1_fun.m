function result = CISE1_fun(A,K,tol,maxit)

%-----------------------------------------------------------------------------------------------------------------------------------------
%%%% CISE algorithm to estimate common structure and low-dimensional individual structure of multiple undirected binary networks #####
%%%------ M-GRAF Model ----------------------------------------------##
% A_i ~ Bernoulli(\Pi_i)
% logit(\Pi_i) = Z + D_i = Z + Q_i %*% \Lambda_i %*% t(Q_i)
% The algorithm iterate between the following steps until convergence
% 1. Given Z, \Lambda_i, solve for Q_i by doing eigen-decomposition on (A_i-P_0), where P_0 = 1/(1+exp(-Z))
% 2. Given Q_i, solve Z and \Lambda_i by logistic regression
%-------------------------------------------------------------------------------------

%---- INPUT ARGUMENTS:
% A: VxVxn array storing n symmetric adjacency matrices (VxV).
% K: latent dimension of each D_i
% tol: convergence threshold for CISE algorithm. CISE iteration continues until the absolute percent change
%       in joint log-likelihood is smaller than this value. Default is 0.01      
% maxit: maximum number of iterations in CISE. Default is 5.

n = size(A,3);
V = size(A,1);
L = V*(V-1)/2;

%% INITIALIZATION 
% initialize P0 by A_bar ------------------------------------
P0 = zeros(V);

A_LT = zeros(L,n);
LTidx = tril(true(V),-1); % lower triangular index

for i=1:n
    A_i = A(:,:,i);
    P0 = P0 + A_i;
    A_LT(:,i) = A_i(LTidx);
end

P0 = P0./n;

% initialize Z by log odds of P0 ---------------------------
vec_P0 = P0(LTidx);
vec_P0(vec_P0==1) = 1-(1e-16);
vec_P0(vec_P0==0) = 1e-16;
Z = log( vec_P0./(1-vec_P0) );

% initialize Lambda_i by eigenvalues of (A_i-P0)------------
% select the first K largest eigenvalues in magnitude
% initialize Q_i by eigenvectors correspond to Lambda_i

Lambda = zeros(K,n);
Q = zeros(V,K,n);
for i=1:n
    % select K largest eigenvalues in magnitude
    [Q(:,:,i),EV] = eigs(A(:,:,i)-P0, K);
    Lambda(:,i) = diag(EV);
end

if nargin < 4 || isempty(tol)
    tol = 0.01;
end

if nargin < 4 || isempty(maxit)
    maxit = 5;
end

cd 'glmnet_matlab'

%% COMPUTE INITIAL LOG-LIKELIHOOD
M_array = zeros(L,K,n);
for i=1:n
    for k=1:K
        M_temp = Q(:,k,i) * Q(:,k,i)';
        M_array(:,k,i) = M_temp(LTidx); % extract lower triangular elements
    end
end

LL_A = 0;

for i=1:n
    vec_Pi = 1./(1 + exp(-Z - M_array(:,:,i) * Lambda(:,i)) );
    vec_Pi(vec_Pi==1) = 1 - (1e-16);
    vec_Pi(vec_Pi==0) = 1e-16;
    LL_A = LL_A + sum( A_LT(:,i).*log(vec_Pi) + (1-A_LT(:,i)).*log(1-vec_Pi) );
end

%% TUNE PENALTY PARAMETER LAMBDA IN GLMNET
% construct y -----------------------------------------
y = A_LT(:);
% construct penalty factors for Z and Lambda ------------
% prior precision of Z
phi_z = 0.01;
% prior precision of lambda
s_l = 2.5;
phi_lambda = 1/s_l^2;
% penalty factor
pen_fac = [phi_z*ones(1,L) phi_lambda*ones(1,n*K)];
% need to ensure sum(pen_fac) = L+n*K, # variables
const_pf = sum(pen_fac)/(L+n*K);
pen_fac = pen_fac./const_pf;
% glmnet penalty factor
lambda_glm = [10.^(0:-1:-8),0].*const_pf;
% construct intercept part of design matrix ---------------
[row_ind, col_ind, val] = find(speye(L));
row_cell = cell(1,n);
col_cell = cell(1,n);
val_cell = cell(1,n);

for i=1:n
    row_cell{i} = row_ind' + (i-1)*L;
    col_cell{i} = col_ind';
    val_cell{i} = val';
end
design_int = sparse([row_cell{:}], [col_cell{:}], [val_cell{:}]);

% construct predictors M part of design matrix -----------------
sd_M = zeros(K,n); % standard deviation (SD)

row_cell = cell(1,n);
col_cell = cell(1,n);
val_cell = cell(1,n);

for i=1:n
    sd_M(:,i) = std(M_array(:,:,i));
    scale_M_i = M_array(:,:,i)./repmat(2.*sd_M(:,i)',L,1);
    [row_ind, col_ind, val] = find(scale_M_i);
    row_cell{i} = row_ind' + (i-1)*L;
    col_cell{i} = col_ind' + (i-1)*K;
    val_cell{i} = val';
end
design_mat = [design_int, sparse([row_cell{:}], [col_cell{:}], [val_cell{:}])];
% spy(design_mat)

% run a penalized likelihood logistic regression (ridge regression)
opts = struct;
opts.alpha = 0;
opts.lambda = lambda_glm;
opts.standardize = false;
opts.intr = false;
opts.penalty_factor = pen_fac;
opts.maxit = 200;
options=glmnetSet(opts);

mglmModel = cvglmnet(design_mat, y,'binomial',options,'deviance',5);
% lambda_opt = mglmModel.lambda_min
ind_lambda_opt = find(lambda_glm == mglmModel.lambda_min);
glm_coef = cvglmnetCoef(mglmModel,'lambda_min'); % first element is 0 (intercept)

% update Z and P0 ----------
Z = glm_coef(2:(L+1)); % Lx1
P0 = zeros(V,V);
P0(LTidx) = 1./(1+exp(-Z));
P0 = P0 + P0';

% update Lambda ------------
Lambda = reshape(glm_coef((L+2):end),K,n); % Kxn
% unscale Lambda
Lambda = Lambda./sd_M/2;
% sort Lambda
Lambda = sort(Lambda,'descend');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2-STEP ITERATIVE ALGORITHM
LL_seq = zeros(1,maxit+1);
LL_seq(1) = LL_A;

% elapse_time = zeros(1,maxit);

for st=1:maxit
    %% update Q 
    Q = zeros(V,K,n);
    
    for i=1:n
        j = sum(Lambda(:,i)>0); % number of lambda >0 for i
        if j==0
            % evals ascend default, need to descend
            [Q(:,K:-1:1,i),~] = eigs(A(:,:,i) - P0, K, 'sa'); 
        elseif j==K
            % evals descend default
            [Q(:,:,i),~] = eigs(A(:,:,i) - P0, K, 'la');
        else
            [Q(:,1:j,i),~] = eigs(A(:,:,i) - P0, j, 'la'); 
            [Q(:,K:-1:(j+1),i), ~] = eigs(A(:,:,i) - P0, K-j, 'sa');
        end
    end
    
    %% compute joint log-likelihood 
    M_array = zeros(L,K,n);
    for i=1:n
        for k=1:K
            M_temp = Q(:,k,i) * Q(:,k,i)';
            M_array(:,k,i) = M_temp(LTidx); % extract lower triangular elements
        end
    end

    LL_A = 0;

    for i=1:n
        vec_Pi = 1./(1 + exp(-Z - M_array(:,:,i) * Lambda(:,i)) );
        vec_Pi(vec_Pi==1) = 1 - (1e-16);
        vec_Pi(vec_Pi==0) = 1e-16;
        LL_A = LL_A + sum( A_LT(:,i).*log(vec_Pi) + (1-A_LT(:,i)).*log(1-vec_Pi) );
    end
    
    LL_seq(st+1) = LL_A;
    
    disp(st)
    
    if LL_seq(st+1) > max(LL_seq(1:st))
        Q_best = Q;
        Lambda_best = Lambda;
        Z_best = Z;
    end
    
    if abs(LL_seq(st+1) - LL_seq(st))/abs(LL_seq(st))< tol
        break
    end
    
    %% CONSTRUCT DESIGN-MATRIX FOR LOGISTIC REGRESSION
    % intercept part of design matrix has been constructed ---------------
    % construct predictors M part of design matrix ------------------------
    sd_M = zeros(K,n); % standard deviation (SD)

    row_cell = cell(1,n);
    col_cell = cell(1,n);
    val_cell = cell(1,n);

    for i=1:n
        sd_M(:,i) = std(M_array(:,:,i));
        scale_M_i = M_array(:,:,i)./repmat(2.*sd_M(:,i)',L,1);
        [row_ind, col_ind, val] = find(scale_M_i);
        row_cell{i} = row_ind' + (i-1)*L;
        col_cell{i} = col_ind' + (i-1)*K;
        val_cell{i} = val';
    end
    design_mat = [design_int, sparse([row_cell{:}], [col_cell{:}], [val_cell{:}])];
    
    %% LOGISTIC REGRESSION
    % penalized logistic regression (ridge regression)
    % Instead of setting penalty = lambda_opt, we use a sequence of larger penalty parameters as warm starts. 
    % This is more robust though may take longer time.

    opts.lambda = lambda_glm(1:ind_lambda_opt);
    options=glmnetSet(opts);

    mglmModel = glmnet(design_mat, y,'binomial',options);
    ind_beta = size(mglmModel.beta,2);
    glm_coef = mglmModel.beta(:,ind_beta);
    
    % update Z and P0 ----------
    Z = glm_coef(1:L); % Lx1
    P0 = zeros(V,V);
    P0(LTidx) = 1./(1+exp(-Z));
    P0 = P0 + P0';

    % update Lambda ------------
    Lambda = reshape(glm_coef((L+1):end),K,n); % Kxn
    % unscale Lambda
    Lambda = Lambda./sd_M/2;
    % sort Lambda
    Lambda = sort(Lambda,'descend');
    
    % elapse_time(st) = toc ;
end

result.Z = Z_best;
result.Lambda = Lambda_best;
result.Q = Q_best;
result.LL_seq = LL_seq;

end