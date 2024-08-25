elastic_net_wrapper_mi <- function(data, outcome=NULL, predictors_con=NULL,predictors_cat=NULL, split=80, outer_cv=NULL, stratified=T,scaling=T,
                                   repeated_cv=1,ensr_cv=10,ensr_alphas=seq(0, 1, length = 10),ensr_lambdas=100,seed=404,
                                   stop_train=NULL,stop_test=NULL,family='binary',pred_min=NULL,pred_max=NULL,m_mice=5,method_mice=NULL,
                                   overall_imputation_strategy='informed',missingness_imputation_strategy = 'xy_train_x_test',
                                   testing_strategy='mixed',MID=T,ncores=NULL,cluster_type="PSOCK",prefer_sensitivity=T){
  
  #### SET-UP ####
  # required packages
  require(ensr)
  require(glmnet)
  require(pROC)
  require(caret)
  require(splitTools)
  require(mice)
  require(parallel)
  require(doParallel)
  `%!in%` = Negate(`%in%`)
  
  # set up cluster
  if (is.null(ncores)==F){
    cluster = makeCluster(ncores, type = cluster_type)
  } else{
    cluster = makeCluster((detectCores() - 1), type = cluster_type)
  }
  registerDoParallel(cl = cluster)
  
  #### SPLIT DATA ####
  
  # combine predictors
  predictors = c(predictors_con,predictors_cat)
  
  # data into y and x
  y = data[outcome]
  x = data[predictors]
  
  # get indices needed for train/test splits
  complete_outcome_indices = which(complete.cases(y))
  complete_predictor_indices = which(complete.cases(x))
  
  # set possible test indices based on settings
  if (missingness_imputation_strategy == 'xy_train_xy_test'|missingness_imputation_strategy == 'x_train_xy_test'){
    possible_test_indices = c(1:nrow(x))
  } else if (missingness_imputation_strategy == 'xy_train_x_test'|missingness_imputation_strategy == 'x_train_x_test'){
    possible_test_indices = c(1:nrow(x))[c(1:nrow(x))%in%complete_outcome_indices]
  } else if (missingness_imputation_strategy == 'xy_train_y_test'|missingness_imputation_strategy == 'x_train_y_test'){
    possible_test_indices = c(1:nrow(x))[c(1:nrow(x))%in%complete_predictor_indices]
  } else if (missingness_imputation_strategy == 'xy_train'|missingness_imputation_strategy == 'x_train'){
    possible_test_indices = c(1:nrow(x))[c(1:nrow(x))%in%complete_predictor_indices&c(1:nrow(x))%in%complete_outcome_indices]
  }
  
  # create list to store the actual test indices to store the predictions
  test_indices_all = list()
  
  # create list of x and y datasets to be analyzed
  split_data_list = list()
  
  # split x and y into training and testing data
  if (is.null(outer_cv)==T){
    # performing stratified split
    if(stratified==T){
      if (family=='binary'){
        # get indices
        set.seed(seed)
        my_train_ind_no_y =  sample(which(unlist(y)[possible_test_indices]==0), size = split/100*length(which(unlist(y)[possible_test_indices]==0)))
        set.seed(seed)
        my_train_ind_y =  sample(which(unlist(y)[possible_test_indices]==1), size = split/100*length(which(unlist(y)[possible_test_indices]==1)))
        test_indices = possible_test_indices[-c(my_train_ind_no_y,my_train_ind_y)]
        # store test indices
        test_indices_all[[1]] = test_indices
        # split data
        y_train = y[-c(test_indices),]
        y_test =y[test_indices,]
        x_train = x[-c(test_indices),]
        x_test = x[test_indices,]
        # add to analysis list
        split_data_list[[1]] = list(y_train,y_test,x_train,x_test)
      }
      else if (family=='continuous'){
        # get indices
        set.seed(seed)
        my_train_ind =  createDataPartition(as.matrix(unlist(y)[possible_test_indices]), p = split/100, list = T,groups=min(3,nrow(y)))
        test_indices = possible_test_indices[-c(unlist(my_train_ind))]
        # store test indices
        test_indices_all[[1]] = test_indices
        # split data
        y_train = y[-c(test_indices),]
        y_test =y[test_indices,]
        x_train = x[-c(test_indices),]
        x_test = x[test_indices,]
        # add to analysis list
        split_data_list[[1]] = list(y_train,y_test,x_train,x_test)
      }
    }
    # performing non-stratified split
    else if(stratified==F){
      set.seed(seed)
      my_train_ind = sample(possible_test_indices, size = split/100*length(possible_test_indices))
      test_indices = possible_test_indices[possible_test_indices%!in%my_train_ind]
      # store test indices
      test_indices_all[[1]] = test_indices
      # split data
      y_train = y[-c(test_indices),]
      y_test =y[test_indices,]
      x_train = x[-c(test_indices),]
      x_test = x[test_indices,]
      # add to analysis list
      split_data_list[[1]] = list(y_train,y_test,x_train,x_test)
    }
  }
  # creating datasets for cross-validation
  else {
    if(stratified==T){
      # creating folds
      set.seed(seed)
      folds <- create_folds(as.numeric(unlist(y)[possible_test_indices]),k = outer_cv,type='stratified')
      # creating datasets
      for(nfold in 1:length(folds)){
        # get indices
        test_indices = possible_test_indices[-c(folds[[nfold]])]
        # store indices
        test_indices_all[[nfold]] = test_indices
        # split data
        y_train <- y[-c(test_indices), ]
        y_test <- y[test_indices, ]
        x_train <- x[-c(test_indices), ]
        x_test <- x[test_indices, ]
        split_data_list[[nfold]] = list(y_train,y_test,x_train,x_test)
        
      }
    }
    else if(stratified==F){
      set.seed(seed)
      folds <- create_folds(as.numeric(unlist(y)[possible_test_indices]),k = outer_cv,type='basic')
      # creating datasets
      for(nfold in 1:length(folds)){
        # get indices
        test_indices = possible_test_indices[-c(folds[[nfold]])]
        # store indices
        test_indices_all[[nfold]] = test_indices
        # split data
        y_train <- y[-c(test_indices), ]
        y_test <- y[test_indices, ]
        x_train <- x[-c(test_indices), ]
        x_test <- x[test_indices, ]
        split_data_list[[nfold]] = list(y_train,y_test,x_train,x_test)
      }
    }
  }
  
  #### IMPUTATION ####
  imputed_data_list = foreach (entry=c(1:length(split_data_list)),.packages = c('mice')) %dopar% {
    
    # create results list
    imputed_data_results = vector("list", length = m_mice)
    
    # getting the training and testing data
    y_train_entry = split_data_list[[entry]][[1]]
    y_test_entry = split_data_list[[entry]][[2]]
    x_train_entry= split_data_list[[entry]][[3]]
    x_test_entry= split_data_list[[entry]][[4]]
    
    # scaling numeric data
    if (scaling==T){
      for(variable in predictors_con){
        mean_variable = mean(as.numeric(unlist(x_train_entry[,variable])),na.rm=T)
        sd_variable = sd(as.numeric(unlist(x_train_entry[,variable])),na.rm=T)
        x_train_entry[,variable] = (as.numeric(unlist(x_train_entry[,variable]))-mean_variable)/sd_variable
        x_test_entry[,variable] = (as.numeric(unlist(x_test_entry[,variable]))-mean_variable)/sd_variable
      }
    }
    
    # removing variables with no variance from the training data
    for (name in colnames(x_train_entry)){
      if (length(unique(na.omit(unlist(x_train_entry[,name]))))<2){
        x_train_entry = x_train_entry[, !colnames(x_train_entry) %in% c(name)]
        x_test_entry = x_test_entry[, !colnames(x_test_entry) %in% c(name)]
      }
    }
    
    # removing collinear columns that mice will not impute
    for (imputation in mice:::find.collinear(x_train_entry)){
      x_train_entry = x_train_entry[, !colnames(x_train_entry) %in% c(name)]
      x_test_entry = x_test_entry[, !colnames(x_test_entry) %in% c(name)]
    }
    
    # identify binary data
    binary_predictors = colnames(x_train_entry)[which(apply(x_train_entry,2,function(x) { all(x %in% 0:1) })==T)]
    binary_predictors = subset(binary_predictors,binary_predictors%!in%colnames(x_train_entry)[grepl('numeric',sapply(x_train_entry,class))])
    
    # transforming to a data matrix
    x_train_entry = data.matrix(x_train_entry)
    x_test_entry = data.matrix(x_test_entry)
    
    # correcting dummy coded variables
    x_train_entry[,c(binary_predictors)]<- x_train_entry[,c(binary_predictors)]-1
    x_test_entry[,c(binary_predictors)]<- x_test_entry[,c(binary_predictors)]-1
    
    # define data to impute if you impute the full dataset
    if (overall_imputation_strategy=='full'){
      # include outcome in imputation
      if (missingness_imputation_strategy == 'xy_train_xy_test'){
        rows_to_ignore = c(rep(FALSE,nrow(x_train_entry)),rep(FALSE,nrow(x_test_entry)))
        data_to_impute = rbind(cbind(x_train_entry,y_train_entry),cbind(x_test_entry,y_test_entry))
      } else if (missingness_imputation_strategy == 'x_train_x_test'){ # do not include outcome in imputation
        rows_to_ignore = c(rep(FALSE,nrow(x_train_entry)),rep(FALSE,nrow(x_test_entry)))
        data_to_impute = rbind(cbind(x_train_entry),cbind(x_test_entry))
      }
    } else if (overall_imputation_strategy=='informed'){ #define data to impute if you use an informed overall strategy where the test data is not used in the imputation of the train data, but is coomplated based on the train data
      # include outcome in imputation
      if (missingness_imputation_strategy == 'xy_train_xy_test'|missingness_imputation_strategy == 'xy_train_x_test'|missingness_imputation_strategy == 'xy_train'){
        rows_to_ignore = c(rep(FALSE,nrow(x_train_entry)),rep(TRUE,nrow(x_test_entry)))
        data_to_impute = rbind(cbind(x_train_entry,y_train_entry),cbind(x_test_entry,y_test_entry))
      } else if (missingness_imputation_strategy == 'x_train_x_test'|missingness_imputation_strategy == 'x_train') { # do not include outcome in imputation
        rows_to_ignore = c(rep(FALSE,nrow(x_train_entry)),rep(TRUE,nrow(x_test_entry)))
        data_to_impute = rbind(x_train_entry,x_test_entry)}
    } else if (overall_imputation_strategy=='stacked'){
      # include outcome in imputation
      if (missingness_imputation_strategy == 'xy_train_xy_test'|missingness_imputation_strategy == 'xy_train_x_test'|missingness_imputation_strategy == 'xy_train'){
        rows_to_ignore = c(rep(FALSE,nrow(x_train_entry)))
        data_to_impute = cbind(x_train_entry,y_train_entry)
        data_to_impute_second = cbind(x_test_entry,y_test_entry)
      } else if (missingness_imputation_strategy == 'x_train_x_test'|missingness_imputation_strategy == 'x_train') { # do not include outcome in imputation
        rows_to_ignore = c(rep(FALSE,nrow(x_train_entry)))
        data_to_impute = x_train_entry
        data_to_impute_second = x_test_entry}
    }
    
    # actual imputation 
    set.seed(seed)
    invisible(capture.output({
      data_imputed = mice(data_to_impute,m=m_mice,method=method_mice,seed=seed,ignore = rows_to_ignore)
    }))
    
    ### Store training data ### 
    
    # get training indices with incomplete outcome
    incomplete_outcome_train_ind = which(is.na(y_train_entry)==T)
    
    # remove those indices with a missing outcome if 'Multiple imputation then delete (MID)' is true
    if (MID==T){
      train_indices_imputed = c(1:nrow(x_train_entry))[c(1:nrow(x_train_entry))%!in%incomplete_outcome_train_ind]
    } else if (MID==F) { # 'Multiple imputation then delete (MID)' is false
      train_indices_imputed = c(1:nrow(x_train_entry))
    }
    
    # loop over imputations
    for (imputation in 1:m_mice){
      
      # get x_train_imputation and y_train_imputation
      if (missingness_imputation_strategy == 'xy_train_xy_test'|missingness_imputation_strategy == 'xy_train_x_test'|missingness_imputation_strategy == 'xy_train'){ # if the outcome was used in the imputation
        y_train_imputation = complete(data_imputed,imputation)[train_indices_imputed,ncol(complete(data_imputed,imputation))]
        x_train_imputation = complete(data_imputed,imputation)[train_indices_imputed,-ncol(complete(data_imputed,imputation))]
      }  else if (missingness_imputation_strategy == 'x_train_x_test'|missingness_imputation_strategy == 'x_train') { # if the outcome was not used in the imputation
        y_train_imputation = y_train_entry[train_indices_imputed]
        x_train_imputation = complete(data_imputed,imputation)[train_indices_imputed,]
      }
      # store y_train_imputation
      imputed_data_results[[imputation]][[1]] = y_train_imputation
      
      # store x_train_imputation
      imputed_data_results[[imputation]][[3]] = x_train_imputation
    }
    
    ### Store testing data ### 
    
    # loop over imputations
    for (imputation in 1:m_mice){
      
      # for an informed imputation strategy
      if (overall_imputation_strategy=='informed'){
        # get x_test_imputation
        if (missingness_imputation_strategy == 'xy_train_xy_test'|missingness_imputation_strategy == 'xy_train_x_test'|missingness_imputation_strategy == 'x_train_x_test'){
          x_test_imputation = complete(data_imputed,imputation)[-c(1:nrow(x_train_entry)),1:ncol(x_test_entry)]
        } else { 
          x_test_imputation = x_test_entry
        }
        # get y_test_imputation
        if (missingness_imputation_strategy == 'xy_train_xy_test'){
          y_test_imputation = complete(data_imputed,imputation)[-c(1:nrow(x_train_entry)),-c(1:ncol(x_test_entry))]
        } else { 
          y_test_imputation = y_test_entry
        }
      } else if (overall_imputation_strategy=='full'){
        # get x_test_imputation
        x_test_imputation = complete(data_imputed,imputation)[-c(1:nrow(x_train_entry)),1:ncol(x_test_entry)]
        # get y_test_imputation
        if (missingness_imputation_strategy == 'xy_train_xy_test'){
          y_test_imputation = complete(data_imputed,imputation)[-c(1:nrow(x_train_entry)),-c(1:ncol(x_test_entry))]
        } else { 
          y_test_imputation = y_test_entry
        }
      } else if (overall_imputation_strategy == 'stacked'){
        
        # create lists to store the data
        x_test_imputation = list()
        y_test_imputation = list()
        
        # set y_test_entry colname to y_train_entry
        colnames(data_to_impute_second)=colnames(complete(data_imputed,imputation))
        
        # define data for the imputation
        data_to_impute = rbind(complete(data_imputed,imputation),data_to_impute_second)
        
        # perform second imputation
        set.seed(seed)
        invisible(capture.output({
          secondary_data_imputed = mice(data_to_impute,m=m_mice,method=method_mice,seed=seed)
        }))
        
        # store test data
        for (secondary_imputation in 1:m_mice){
          if (missingness_imputation_strategy == 'xy_train_xy_test'|missingness_imputation_strategy == 'xy_train_x_test'|missingness_imputation_strategy == 'xy_train'){
            y_test_imputation[[secondary_imputation]] = complete(secondary_data_imputed,secondary_imputation)[-c(1:nrow(complete(data_imputed,imputation))),ncol(complete(data_imputed,imputation))]
            x_test_imputation[[secondary_imputation]] = complete(secondary_data_imputed,secondary_imputation)[-c(1:nrow(complete(data_imputed,imputation))),-ncol(complete(data_imputed,imputation))]
          } else if (missingness_imputation_strategy == 'x_train_x_test'|missingness_imputation_strategy == 'x_train') {
            y_test_imputation[[secondary_imputation]] = y_test_entry
            x_test_imputation[[secondary_imputation]] = complete(secondary_data_imputed,secondary_imputation)[-c(1:nrow(complete(data_imputed,imputation))),]
          }
        }
      }
      imputed_data_results[[imputation]][[4]] = x_test_imputation
      imputed_data_results[[imputation]][[2]] = y_test_imputation
    }
    
    # return results
    return(imputed_data_results)
  }
  
  #### FIT MODELS ####
  results_model_fitting = foreach (entry=c(1:length(split_data_list))) %:%  foreach(imputation = 1:m_mice,.packages = c('mice','glmnet','ensr')) %dopar% {
    
    # create results list
    results_imputation = list()
    
    # getting the training and testing data
    y_train_imputation = imputed_data_list[[entry]][[imputation]][[1]]
    x_train_imputation= imputed_data_list[[entry]][[imputation]][[3]]
    
    # Stopping if there aren't enough observations in the training data
    if (is.null(stop_train)==F){
      if (sum(as.numeric(as.character(unlist(y_train_imputation))),na.rm=T)<stop_train){return(results_imputation)}
    }
    
    # find best elastic net model for each imputed data set
    # finding best lambda and alpha
    
    # creating a variable for storing the crossvalidation results for the alphas and the lambdas
    MSEs = NULL
    
    # store variables for  ensr
    x_train_imputation <<- x_train_imputation
    y_train_imputation <<- y_train_imputation
    ensr_lambdas <<- ensr_lambdas
    ensr_cv <<- ensr_cv
    ensr_alphas <<- ensr_alphas
    
    # get ensr family
    ensr_family <<- ifelse(family=='binary','binomial','gaussian')
    
    for (repeated_cv_number in 1:repeated_cv){
      
      # setting the seed
      set.seed(repeated_cv_number)
      # selecting the best alpha and lambda for this seed
      ensr_obj = ensr(y =data.matrix(y_train_imputation), x = data.matrix(x_train_imputation),nlambda=ensr_lambdas,nfolds = ensr_cv,
                      alphas = ensr_alphas,family=ensr_family,standardize = F)
      ensr_obj_summary = summary(object = ensr_obj)
      
      # storing the results
      MSEs = cbind(MSEs,ensr_obj_summary$cvm)
    }
    
    # converting the cross validation results to a dataframe
    MSEs = as.data.frame(MSEs)
    MSEs$rowMeans = rowMeans(MSEs)
    
    # adding the alphas and lambdas that we used
    # these are the same for every seed!
    MSEs$lambdas = ensr_obj_summary$lambda
    MSEs$alphas= ensr_obj_summary$alpha
    MSEs = MSEs[order(MSEs$rowMeans,decreasing = F), ]
    
    # Selecting the  alpha and the lambda of the best model
    alpha.min = MSEs$alphas[1]
    lambda.min = MSEs$lambdas[1]
    
    # fitting the elastic net model and getting the estimates for the variables
    elastic_model = glmnet(y =data.matrix(y_train_imputation), x = x_train_imputation, family = ensr_family, alpha = alpha.min,
                           lambda=lambda.min,standardize = F)
    estimates = elastic_model$beta
    
    # having at least one parameter
    while (length(which(estimates!=0))<1){
      MSEs = MSEs[-1,]
      lambda.min = MSEs$lambdas[1]
      alpha.min = MSEs$alphas[1]
      elastic_model = glmnet(y =data.matrix(y_train_imputation), x = x_train_imputation, family = ensr_family,
                             alpha = alpha.min,lambda=lambda.min,standardize = F)
      estimates = elastic_model$beta
    }
    
    # store estimates
    estimates_imputation = data.frame(matrix(ncol=(ncol(x_train_imputation)+1),nrow=1))
    colnames(estimates_imputation) = c('intercept',colnames(x_train_imputation))
    estimates_imputation[1,1] = elastic_model$a0
    estimates_imputation[1,2:ncol(estimates_imputation)] = estimates
    results_imputation[[1]] = estimates_imputation
    
    # store model
    results_imputation[[2]] = elastic_model
    
    # return results
    return(results_imputation)
  }
  
  #### CALCULATE METRICS ####
  
  # calculate metrics for cross-validation every fold
  metrics = foreach (entry = c(1:length(split_data_list)),.combine = 'rbind') %:% foreach(imputation = 1:m_mice,.combine = 'rbind') %:% foreach(test_data_entry = 1:m_mice,.packages = c('pROC','caret'),.combine = 'rbind') %dopar% {
    
    # calculate metrics for a binary outcome
    if (family=='binary') {
      
      # create results df 
      results_df = data.frame(matrix(ncol = (11+length(predictors))))
      colnames(results_df) = c('fold','imputation','test_data_entry','nrow_train','nrow_test','AUC','sensitivity','specificity',
                               'accuracy','PPV','NPV',predictors)
      
      # skip if there are no results
      if (length(results_model_fitting[[entry]][[imputation]])==0){return(results_df)}
      
      # skip if the strategy is matched and the imputation number does not match the test data number
      if (overall_imputation_strategy=='full'|testing_strategy=='matched'){
        if(imputation!=test_data_entry){
          return(results_df)}}
      
      # get the estimates for the imputation
      estimates = as.numeric(results_model_fitting[[entry]][[imputation]][[1]])
      
      # predict test data
      if (overall_imputation_strategy=='full'|overall_imputation_strategy=='informed'){
        predictions_imputation = data.matrix(imputed_data_list[[entry]][[test_data_entry]][[4]])%*%estimates[2:length(estimates)]+estimates[1]
      } else if (overall_imputation_strategy=='stacked') { # predict each test data from the secondary imputation
        predictions_imputation = matrix(ncol=m_mice,nrow=nrow(imputed_data_list[[entry]][[test_data_entry]][[4]][[secondary_imputation]]))
        for (secondary_imputation in 1:m_mice){
          predictions_secondary_imputation = data.matrix(imputed_data_list[[entry]][[test_data_entry]][[4]][[secondary_imputation]])%*%estimates[2:length(estimates)]+estimates[1]
          predictions_imputation[,secondary_imputation]=predictions_secondary_imputation
        }
        predictions_imputation = rowMeans(predictions_imputation)
      }
      
      # take the exponential of the predictions
      predictions_imputation = exp(predictions_imputation)/(1+exp(predictions_imputation))
      
      # store descriptives
      results_df[1,'fold']=entry
      results_df[1,'imputation']=imputation
      results_df[1,'test_data_entry']=test_data_entry
      results_df[1,'nrow_train']=nrow(split_data_list[[entry]][[3]])
      results_df[1,'nrow_test']=nrow(split_data_list[[entry]][[4]])
      
      # Stopping if there aren't enough positive observations in the training data
      if (is.null(stop_test)==F){
        if (sum(as.numeric(as.character(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]]))),na.rm=T)<stop_test){return(results_df)}
      }
      
      # calculate the actual metrics
      if (overall_imputation_strategy=='full'|overall_imputation_strategy=='informed'){
        
        # AUC
        model_roc =  roc(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]]),as.numeric(predictions_imputation),direction="<",quiet=T)
        model_coords = coords(model_roc,"best", ret=c("threshold", "specificity", "sensitivity"), transpose=FALSE)
        model_auc = auc(model_roc)
        
        # Sensitivity and specificity
        if (prefer_sensitivity==T){
          coords_to_pick = which(model_coords$sensitivity==max(model_coords$sensitivity))
        }
        else {
          coords_to_pick = which(model_coords$specificity==max(model_coords$specificity))
        }
        model_spec <- model_coords[coords_to_pick,2]
        model_sens <- model_coords[coords_to_pick,3]
        
        # accuracy, PPV, NPV
        predictions_bin = ifelse(predictions_imputation>model_coords$threshold[coords_to_pick],1,0)
        confmatrix = confusionMatrix(as.factor(predictions_bin),as.factor(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]])),positive='1')
        accuracy = confmatrix$overall[1]
        PPV = confmatrix$byClass[3]
        NPV = confmatrix$byClass[4]
        
        
      } else if (overall_imputation_strategy=='stacked') {
        
        # create vectors to store the results
        model_auc = c()
        model_sens = c()
        model_spec = c()
        accuracy = c()
        PPV = c()
        NPV = c()
        
        for (secondary_imputation in 1:m_mice){
          
          # AUC, sensitivity, specificity
          model_roc_secondary =  roc(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]][[secondary_imputation]]),as.numeric(predictions_imputation),direction="<",quiet=T)
          model_coords_secondary = coords(model_roc_secondary,"best", ret=c("threshold", "specificity", "sensitivity"), transpose=FALSE)
          model_auc_secondary = auc(model_roc_secondary)
          ifelse( nrow(model_coords_secondary[2])>1,model_spec_secondary <- NA, model_spec_secondary <- model_coords_secondary[2])
          ifelse( nrow(model_coords_secondary[2])>1,model_sens_secondary <- NA, model_sens_secondary <- model_coords_secondary[3])
          
          # accuracy, PPV, NPV
          predictions_bin_secondary = ifelse(predictions_imputation>model_coords_secondary$threshold,1,0)
          confmatrix_secondary = confusionMatrix(as.factor(predictions_bin_secondary),as.factor(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]][[secondary_imputation]])),positive='1')
          
          # store results
          model_auc[secondary_imputation] = model_auc_secondary
          model_sens[secondary_imputation] = as.numeric(model_sens_secondary)
          model_spec[secondary_imputation] = as.numeric(model_spec_secondary)
          accuracy[secondary_imputation] = confmatrix_secondary$overall[1]
          PPV[secondary_imputation] = confmatrix_secondary$byClass[3]
          NPV[secondary_imputation] = confmatrix_secondary$byClass[4]
        }
        
        model_auc = mean(model_auc)
        model_sens = mean(model_sens)
        model_spec = mean(model_spec)
        accuracy = mean(accuracy)
        PPV = mean(PPV)
        NPV = mean(NPV)
      }
      
      # store metrics
      results_df[1,'AUC']=model_auc
      results_df[1,'sensitivity']=model_sens
      results_df[1,'specificity']=model_spec
      results_df[1,'accuracy']=accuracy
      results_df[1,'PPV']=PPV
      results_df[1,'NPV']=NPV
      
      # storing estimates
      for (predictor in predictors){
        index = which(names(results_model_fitting[[entry]][[imputation]][[1]])==predictor)
        if (length(index)==0){
          results_df[1,predictor]<- NA
        }
        else{
          results_df[1,predictor]<- results_model_fitting[[entry]][[imputation]][[1]][index]
        }
      }  
    } else if (family=='continuous'){
      
      # create results df 
      results_df = data.frame(matrix(ncol = (10+length(predictors))))
      colnames(results_df) = c('fold','imputation','test_data_entry','nrow_train','nrow_test','R2','R2_adjusted','RMSE','MSE',
                               'MAE',predictors)
      
      # skip if there are no results
      if (length(results_model_fitting[[entry]][[imputation]])==0){return(results_df)}
      
      # skip if the strategy is matched and the imputation number does not match the test data number
      if (overall_imputation_strategy=='full'|testing_strategy=='matched'){
        if(imputation!=test_data_entry){
          return(results_df)}}
      
      # get the estimates for the imputation
      estimates = as.numeric(results_model_fitting[[entry]][[imputation]][[1]])
      
      # predict test data
      if (overall_imputation_strategy=='full'|overall_imputation_strategy=='informed'){
        predictions_imputation = data.matrix(imputed_data_list[[entry]][[test_data_entry]][[4]])%*%estimates[2:length(estimates)]+estimates[1]
      } else if (overall_imputation_strategy=='stacked') {
        predictions_imputation = matrix(ncol=m_mice,nrow=nrow(imputed_data_list[[entry]][[test_data_entry]][[4]][[secondary_imputation]]))
        for (secondary_imputation in 1:m_mice){
          predictions_secondary_imputation = data.matrix(imputed_data_list[[entry]][[test_data_entry]][[4]][[secondary_imputation]])%*%estimates[2:length(estimates)]+estimates[1]
          predictions_imputation[,secondary_imputation]=predictions_secondary_imputation
        }
        predictions_imputation = rowMeans(predictions_imputation)
      }
      
      # store descriptives
      results_df[1,'fold']=entry
      results_df[1,'imputation']=imputation
      results_df[1,'test_data_entry']=test_data_entry
      results_df[1,'nrow_train']=nrow(split_data_list[[entry]][[3]])
      results_df[1,'nrow_test']=nrow(split_data_list[[entry]][[4]])
      
      # adapt predictions
      if (is.null(pred_min)==F){predictions_imputation[predictions_imputation<pred_min]=pred_min}
      if (is.null(pred_max)==F){predictions_imputation[predictions_imputation>pred_max]=pred_max}
      
      # calculate the actual metrics
      if (overall_imputation_strategy=='full'|overall_imputation_strategy=='informed'){
        
        # Getting R2, adjusted R2, RMSE, MSE, MAE
        R2 = 1-(sum((imputed_data_list[[entry]][[test_data_entry]][[2]]-predictions_imputation)^2)/length(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]])))/(sum((imputed_data_list[[entry]][[test_data_entry]][[2]]-mean(unlist(imputed_data_list[[entry]][[imputation]][[1]])))^2)/length(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]])))
        R2_adjusted = 1 - (1-R2)*(length(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]]))-1)/(length(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]]))-length(which(estimates!=0))-1)
        RMSE = RMSE(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]]),predictions_imputation)
        MSE = RMSE^2
        MAE = MAE(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]]),predictions_imputation)
        
      } else if (overall_imputation_strategy=='stacked') {
        
        # create vectors to store the results
        R2 = c()
        R2_adjusted = c()
        RMSE = c()
        MSE = c()
        MAE = c()
        
        for (secondary_imputation in 1:m_mice){
          
          # Getting R2, adjusted R2, RMSE, MSE, MAE
          R2_secondary = 1-(sum((imputed_data_list[[entry]][[test_data_entry]][[2]][[secondary_imputation]]-predictions_imputation)^2)/length(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]][[secondary_imputation]])))/(sum((imputed_data_list[[entry]][[test_data_entry]][[2]][[secondary_imputation]]-mean(unlist(imputed_data_list[[entry]][[imputation]][[1]])))^2)/length(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]][[secondary_imputation]])))
          R2_adjusted_secondary = 1 - (1-R2_secondary)*(length(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]][[secondary_imputation]]))-1)/(length(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]][[secondary_imputation]]))-length(which(estimates!=0))-1)
          RMSE_secondary = RMSE(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]][[secondary_imputation]]),predictions_imputation)
          MSE_secondary = RMSE_secondary^2
          MAE_secondary = MAE(unlist(imputed_data_list[[entry]][[test_data_entry]][[2]][[secondary_imputation]]),predictions_imputation)
          
          # store results
          R2[secondary_imputation] = R2_secondary
          R2_adjusted[secondary_imputation] = R2_adjusted_secondary
          RMSE[secondary_imputation] = RMSE_secondary
          MSE[secondary_imputation] = MSE_secondary
          MAE[secondary_imputation] = MAE_secondary
        }
        
        R2 = mean(R2)
        R2_adjusted = mean(R2_adjusted)
        RMSE = mean(RMSE)
        MSE = mean(MSE)
        MAE = mean(MAE)
      }
      
      # storing metrics
      results_df[1,'R2']=R2
      results_df[1,'R2_adjusted']=R2_adjusted
      results_df[1,'RMSE']=RMSE
      results_df[1,'MSE']=MSE
      results_df[1,'MAE']=MAE
      
      # storing estimates
      for (predictor in predictors){
        index = which(names(results_model_fitting[[entry]][[imputation]][[1]])==predictor)
        if (length(index)==0){
          results_df[1,predictor]<- NA
        }
        else{
          results_df[1,predictor]<- results_model_fitting[[entry]][[imputation]][[1]][index]
        }
      }  
    }
    return(results_df)
  }
  
  # aggregate results
  metrics = aggregate(metrics[,-c(1,2,3)],list(fold=metrics$fold),mean,na.rm=T)
  
  # Create final results
  results = list()
  results$metrics = metrics
  results$results_model_fitting = results_model_fitting
  
  # stop cluster
  stopCluster(cl = cluster)
  
  # return results
  return(results)
}
