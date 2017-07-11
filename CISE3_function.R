#######################################################################################################################################
##### CISE algorithm to estimate common structure and low-dimensional individual structure of multiple undirected binary networks #####
##------- M-GRAF Model, Variant 3 -----------------------------------##
## A_i ~ Bernoulli(\Pi_i)
## logit(\Pi_i) = Z + D_i = Z + Q %*% \Lambda_i %*% t(Q)
## Q is the same for all subjects
## The algorithm iterate between the following steps until convergence
## 1. Given Z, \Lambda_i, solve for Q by seting each column k of Q at the first eigenvector of
##    sum_{i=1}^{n} [\lambda_{ik}(A_i-P_0)] corresponding to the largest eigenvalue, 
##    where P_0 = 1/(1+exp(-Z))
## 2. Given Q, solve Z and \Lambda_i by logistic regression
#######################################################################################################################################

library(gdata)
library(Matrix)
library(glmnet)
library(rARPACK)
library(MASS)
library(far)

# res = CISE3(A=A, K=5, tol=0.01, maxit=5)

CISE3 = function(A, K, tol, maxit){
  # A: VxVxn array storing n symmetric adjacency matrices (VxV).
  # K: latent dimension of each D_i
  # tol: convergence threshold for CISE algorithm. CISE iteration continues until the absolute percent change
  #       in joint log-likelihood is smaller than this value. Default is 0.01      
  # maxit: maximum number of iterations in CISE. Default is 5.

  n=dim(A)[3]
  V=dim(A)[1]
  L = V*(V-1)/2
  
  if(missing(tol)){
    tol = 0.01
  }
  
  if(missing(maxit)){
    maxit = 5
  }
  ######################################################################################
  ###### Initialization ----------------------------------------------------------------
  
  ## initialize P_0 by A_bar -----------------------------------------------------------
  P0 = apply(A,c(1,2),sum)/n
  ## initialize Z by log odds of P0 (much faster than using logistic regression)
  vec_P0 = lowerTriangle(P0)
  vec_P0[which(vec_P0==1)] = 1 - (1e-16) # note 1 - (1-(1e-17)) == 0 in R
  vec_P0[which(vec_P0==0)] = 1e-16
  Z = log(vec_P0/(1-vec_P0))
  
  ## initialize Lambda_i by eigenvalues of (A_i-P0) ------------------------------------ 
  Lambda = matrix(0,nr=K,nc=n)
  ## select the first K largest eigenvalue *in magnitude*
  for (i in 1:n){
    ED = eigs_sym(A = A[,,i]-P0, k=K, which="LM", opts = list(retvec = FALSE))
    Lambda[,i] = ED$values
  }
  #########################################################################################
  ################################# SOLVE Q -----------------------------------------------
  # create intermediate matrices W_k = = sum_{i=1}^n {lambda_{ik}* [A_i - P0] } -------------------
  W_array = array(0, c(V,V,K)) 
  
  for(k in 1:K){
    temp = sweep(A, 3, Lambda[k,], FUN="*")
    temp = apply(temp, c(1,2), sum) # temp = sum_{i=1}^n [lambda_{ik}*A_i]
    # temp = sum_{i=1}^n [lambda_{ik}*A_i] - (sum_i^n (lambda_{ik}) * P0
    W_array[,,k] = temp - sum(Lambda[k,]) * P0 
  }
  
  Q = matrix(0,nr=V,nc=K)
  
  ## find the first column of Q to update -----------------------------------------------------
  s1 = which.max( apply(W_array, 3, function(W){
    eigs_sym(W, k=1, which = "LA", opts = list(retvec = FALSE))$values
  }) )
  
  ED1 = eigen(W_array[,,s1], symmetric = T)
  Q[,s1] = ED1$vec[,1]
  
  if(K>=2){
    det_col = c(s1)
    rem_col = setdiff(1:K, s1)
    ## create orthogonal basis that's perpendicular to Q[,s1]
    U = ED1$vec[,-1]
    
    #### find the next col of Q to update -----------------------------------------------------
    s2 = which.max( apply(W_array[,,rem_col], 3, function(W){
      eigs_sym(t(U) %*% W %*% U, k=1, which="LA", opts = list(retvec = FALSE))$values
    }) )
    s2 = rem_col[s2]
    
    a = eigs_sym(t(U) %*% W_array[,,s2] %*% U, k=1, which="LA")$vectors
    Q[,s2] = U %*% a
    
    if(K>=3){
      for(k in 3:K){
        det_col = c(det_col, s2)
        rem_col = setdiff(rem_col, s2)
        ## update ortho basis to perp to previous determined columns of Q
        U_temp = orthonormalization(u=Q[,det_col], basis = T, norm=T)
        U = U_temp[,k:V]
        
        if(k==V){
          Q[,rem_col] = U
        }else{
          s2 = which.max( apply(W_array[,,rem_col, drop=F], 3, function(W){
            eigs_sym(t(U) %*% W %*% U, k=1, which="LA", opts = list(retvec = FALSE))$values
          }) )
          s2 = rem_col[s2]
          
          a = eigs_sym(t(U) %*% W_array[,,s2] %*% U,  k=1, which="LA")$vectors
          Q[,s2] = U %*% a
          
        }
      }
    }
  }
  #########################################################################################
  #### compute joint log-likelihood --------------------------------------------------
  ## convert each lower-triangular adjacency matrix to a vector
  A_LT = apply(A,3,lowerTriangle)
  
  ## an array of lower-triangle of principal component matrices for computation convenience
  M = apply(Q, 2, function(x){lowerTriangle( x %*% t(x) )}) # LxK
  
  LL_A = 0
  
  for(i in 1:n){
    vec_Pi = 1/(1 + exp(-Z - M %*% Lambda[,i]) )
    vec_Pi[which(vec_Pi==1)] = 1 - (1e-16) # note 1 - (1-(1e-17)) == 0 in R
    vec_Pi[which(vec_Pi==0)] = 1e-16
    LL_A = LL_A + sum( A_LT[,i]*log(vec_Pi) + (1-A_LT[,i])*log(1-vec_Pi) ) 
  }
  
  ######################################################################################
  ####### TUNE PENALTY PARAMETER LAMBDA IN GLMNET
  ### CONSTRUCT Y ----------------------------------------------------------------------
  y = factor(c(A_LT))
  
  ### CONSTRUCT PENALTY FACTORS FOR Z AND LAMBDA ---------------------------------------
  # prior precision of Z
  phi_z = 0.01
  # prior precision of lambda
  s_l = 2.5 # prior scale
  phi_lambda = 1/(s_l^2) 
  # penalty factor
  pen_fac = c(rep(phi_z,L), rep(phi_lambda,n*K))
  # normalize to ensure sum(pen_fac) = L+n*K, #variables
  const_pf = sum(pen_fac)/(L+n*K)
  pen_fac = pen_fac/const_pf
  # glmnet penalty factor
  lambda_glm = c(10^(0:-8),0)*const_pf
  
  ### CONSTRUCT DESIGN-MATRIX ----------------------------------------------------------
  ## construct intercept part of design matrix -----------------------------------------
  design_int = Diagonal(L)
  for(i in 2:n){
    design_int = rbind(design_int, Diagonal(L))
  }
  
  ## construct predictors M part of design matrix --------------------------------------
  ## scale M
  sd_M = apply(M, 2, sd) # Kx1
  M_list = lapply(1:n, function(i){
    # scale M[,k] to have sd 0.5
    temp_M = sweep(M, 2, 2*sd_M, FUN = "/") # LxK
  })
  
  design_mat = cbind(design_int, bdiag(M_list))
  rm(M_list)
  
  ## run cv.glmnet to determine optimal penalty lambda
  rglmModel = cv.glmnet(x=design_mat, y=y, family="binomial", alpha=0, lambda=lambda_glm, 
                        standardize = FALSE, intercept = FALSE, penalty.factor = pen_fac, 
                        maxit=200, nfolds = 5, parallel = FALSE) # type.measure=deviance 
  ind_lambda_opt = which(lambda_glm == rglmModel$lambda.min)
  glm_coef = coef(rglmModel, s="lambda.min")[-1]
  
  ##----- update Z and P0 -----##
  Z = glm_coef[1:L] # Lx1
  P0 = matrix(0,nr=V,nc=V)
  lowerTriangle(P0) = 1/(1+exp(-Z))
  P0 = P0 + t(P0)
  
  ##----- update Lambda -----##
  Lambda = matrix(glm_coef[(L+1):(L+n*K)], nrow=K, ncol=n) # Kxn
  # unscale Lambda
  Lambda = sweep(Lambda, 1, 2*sd_M, FUN = "/")
  
  ########################################################################################
  ### 2-step Iterative Algorithm ---------------------------------------------------------
  ########################################################################################
  LL_seq = numeric(maxit + 1)
  LL_seq[1] = LL_A
  
  # elapse_time = numeric(maxit)
  for (st in 1:maxit){
    ptm=proc.time()
    #########--- Update Q --------------------------------------------------------------
    # create intermediate matrices W_k = = sum_{i=1}^n {lambda_{ik}* [A_i - P0] } -------------------
    W_array = array(0, c(V,V,K)) 
    
    for(k in 1:K){
      temp = sweep(A, 3, Lambda[k,], FUN="*")
      temp = apply(temp, c(1,2), sum) # temp = sum_{i=1}^n [lambda_{ik}*A_i]
      # temp = sum_{i=1}^n [lambda_{ik}*A_i] - (sum_i^n (lambda_{ik}) * P0
      W_array[,,k] = temp - sum(Lambda[k,]) * P0 
    }
    
    Q = matrix(0,nr=V,nc=K)
    
    ## find the first column of Q to update -----------------------------------------------------
    s1 = which.max( apply(W_array, 3, function(W){
      eigs_sym(W, k=1, which = "LA", opts = list(retvec = FALSE))$values
    }) )
    
    ED1 = eigen(W_array[,,s1], symmetric = T)
    Q[,s1] = ED1$vec[,1]
    
    if(K>=2){
      det_col = c(s1)
      rem_col = setdiff(1:K, s1)
      ## create orthogonal basis that's perpendicular to Q[,s1]
      U = ED1$vec[,-1]
      
      #### find the next col of Q to update -----------------------------------------------------
      s2 = which.max( apply(W_array[,,rem_col], 3, function(W){
        eigs_sym(t(U) %*% W %*% U, k=1, which="LA", opts = list(retvec = FALSE))$values
      }) )
      s2 = rem_col[s2]
      
      a = eigs_sym(t(U) %*% W_array[,,s2] %*% U, k=1, which="LA")$vectors
      Q[,s2] = U %*% a
      
      if(K>=3){
        for(k in 3:K){
          det_col = c(det_col, s2)
          rem_col = setdiff(rem_col, s2)
          ## update ortho basis to perp to previous determined columns of Q
          U_temp = orthonormalization(u=Q[,det_col], basis = T, norm=T)
          U = U_temp[,k:V]
          
          if(k==V){
            Q[,rem_col] = U
          }else{
            s2 = which.max( apply(W_array[,,rem_col, drop=F], 3, function(W){
              eigs_sym(t(U) %*% W %*% U, k=1, which="LA", opts = list(retvec = FALSE))$values
            }) )
            s2 = rem_col[s2]
            
            a = eigs_sym(t(U) %*% W_array[,,s2] %*% U,k=1, which="LA")$vectors
            Q[,s2] = U %*% a
            
          }
        }
      }
    }
    
    #### compute joint log-likelihood --------------------------------------------------
    ## an array of lower-triangle of principal component matrices for computation convenience
    M = apply(Q, 2, function(x){lowerTriangle( x %*% t(x) )}) # LxK
    
    LL_A = 0
    
    for(i in 1:n){
      vec_Pi = 1/(1 + exp(-Z - M %*% Lambda[,i]) )
      vec_Pi[which(vec_Pi==1)] = 1 - (1e-16) # note 1 - (1-(1e-17)) == 0 in R
      vec_Pi[which(vec_Pi==0)] = 1e-16
      LL_A = LL_A + sum( A_LT[,i]*log(vec_Pi) + (1-A_LT[,i])*log(1-vec_Pi) ) 
    }
    
    LL_seq[st+1] = LL_A
    
    print(st)
    
    if(LL_seq[st+1] > max(LL_seq[1:st])){
      Q_best = Q
      Lambda_best = Lambda
      Z_best = Z
      LL_max = LL_seq[st+1]
    }
    
    ############ CONSTRUCT DESIGN-MATRIX FOR LOGISTIC REGRESSION ----------------------------------
    ## intercept part of design matrix has been constructed --------------------------------------
    ## construct predictors M part of design matrix --------------------------------------
    ## scale M
    sd_M = apply(M, 2, sd) # Kx1
    M_list = lapply(1:n, function(i){
      # scale M[,k] to have sd 0.5
      temp_M = sweep(M, 2, 2*sd_M, FUN = "/") # LxK
    })
    
    design_mat = cbind(design_int, bdiag(M_list))
    rm(M_list)
    
    ########## LOGISTIC REGRESSION ------------------------------------------------------------
    ## run a penalized logistic regression (ridge regression)
    ## Instead of setting penalty = lambda_opt, we use a sequence of larger penalty parameters as warm starts. 
    ## This is more robust though may take longer time.
    
    rglmModel = glmnet(x=design_mat, y=y, family="binomial", alpha=0, lambda=lambda_glm[1:ind_lambda_opt], 
                       standardize = FALSE, intercept = FALSE, penalty.factor = pen_fac, maxit=200)  
    
    ind_beta = dim(rglmModel$beta)[2] # the last column corresponds to the optimal penalty lambda
    ##----- update Z and P0 -----##
    Z = rglmModel$beta[1:L, ind_beta] # Lx1
    P0 = matrix(0,nr=V,nc=V)
    lowerTriangle(P0) = 1/(1+exp(-Z))
    P0 = P0 + t(P0)
    
    ##----- update Lambda -----##
    Lambda = matrix(rglmModel$beta[(L+1):(L+n*K), ind_beta], nrow=K, ncol=n) # Kxn
    # unscale Lambda
    Lambda = Lambda/sd_M/2
    
    # elapse_time[st] = as.numeric((proc.time()-ptm))[3]
  }
  
  results = list(Z=Z_best, Lambda = Lambda_best, Q = Q_best, 
                 LL_max = LL_max, LL = LL_seq)
  # Z: lower triangular elements in the estimated Z
  # Lambda: Kx1 vector; the diagonal entries in \Lambda
  # Q: VxK matrix
  # LL_max: the maximum log-likelihood across the iterations
  # LL: log-likelihood at each iteration
  return(results)
  
}  