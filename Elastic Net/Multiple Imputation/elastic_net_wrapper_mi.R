elastic_net_wrapper_mi <- function(data, outcome=NULL, predictors_con=NULL,predictors_cat=NULL, split=80, outer_cv=NULL, stratified=T,scaling=T,
                                repeated_cv=1,ensr_cv=10,ensr_alphas=seq(0, 1, length = 10),ensr_lambdas=100,seed=404,shuffle=T,
                                stop_train=NULL,stop_test=NULL,family='binary',pred_min=NULL,pred_max=NULL,
                                include_missing_outcome_train=T,include_missing_predictors_test=T,
                                include_missing_outcome_test=F,include_outcome_imputation = T,
                                m_mice=5,method_mice=NULL,MID=T){
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
  
  # set up cores
  cluster = makeCluster((detectCores() - 1), type = "PSOCK")
  registerDoParallel(cl = cluster)
  
  # combine predictors
  predictors = c(predictors_con,predictors_cat)
  
  # shuffle dataset to lose time contingency for CV
  if (shuffle==T){
    set.seed(seed)
    data = data[sample(nrow(data)),]
  }
  
  # reduce data if missingness in the outcome in not accepted
  if (include_missing_outcome_train==F){
    data = data[complete.cases(data[outcome]),]
  }
  
  # data into y and x
  y = data[outcome]
  x = data[predictors]
  
  # get indices needed for train/test splits
  complete_outcome_indices = which(complete.cases(y))
  complete_predictor_indices = which(complete.cases(x))
  
  # set possible test indices based on settings
  if (include_missing_predictors_test==T&include_missing_outcome_test==T){
    possible_test_indices = c(1:nrow(x))
  } else if (include_missing_predictors_test==T&include_missing_outcome_test==F){
    possible_test_indices = c(1:nrow(x))[c(1:nrow(x))%in%complete_outcome_indices]
  } else if (include_missing_predictors_test==F&include_missing_outcome_test==T){
    possible_test_indices = c(1:nrow(x))[c(1:nrow(x))%in%complete_predictor_indices]
  } else if (include_missing_predictors_test==F&include_missing_outcome_test==F){
    possible_test_indices = c(1:nrow(x))[c(1:nrow(x))%in%complete_predictor_indices&c(1:nrow(x))%in%complete_outcome_indices]
  }
    
  # create list to store the predictions
  predictions_all = rep('Train',length(unlist(y)))
  
  # create list to store the actual test indices to store the predictions
  test_indices_all = list()
  
  # create list of x and y datasets to be analyzed
  analysis_list = list()
  
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
        analysis_list[[1]] = list(y_train,y_test,x_train,x_test)
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
        analysis_list[[1]] = list(y_train,y_test,x_train,x_test)
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
      analysis_list[[1]] = list(y_train,y_test,x_train,x_test)
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
        analysis_list[[nfold]] = list(y_train,y_test,x_train,x_test)
        
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
        analysis_list[[nfold]] = list(y_train,y_test,x_train,x_test)
      }
    }
  }
  
  # creating the results dataframe
  if (family==('binary')){
    results_df = data.frame(matrix(ncol = (11+length(predictors))))
    colnames(results_df) = c('fold','nrow_train','nrow_test','ny_train','ny_test','AUC','sensitivity','specificity',
                             'accuracy','PPV','NPV',predictors)
  }
  if (family==('continuous')){
    results_df = data.frame(matrix(ncol = (10+length(predictors))))
    colnames(results_df) = c('fold','nrow_train','nrow_test','mean_y_train','mean_y_test','R2','R2_adjusted','RMSE','MSE',
                             'MAE',predictors)
  }
  
  # Create progress bar
  print('Training and evaluating the models')
  pb = txtProgressBar(min = 0, max = length(analysis_list), initial = 0) 
  
  # Training and testing the elastic net
  for (entry in 1:length(analysis_list)){
    
    # getting the training and testing data
    y_train_entry = analysis_list[[entry]][[1]]
    y_test_entry= analysis_list[[entry]][[2]]
    x_train_entry= analysis_list[[entry]][[3]]
    x_test_entry= analysis_list[[entry]][[4]]
    
    # Stopping if there aren't enough observations in the training data
    if (is.null(stop_train)==F){
      if (sum(as.numeric(as.character(unlist(y_train_entry))),na.rm=T)<stop_train){next}
    }
    
    #scaling numeric data
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
      if (length(unique(unlist(x_train_entry[,name])))<2){
        x_train_entry = x_train_entry[, !colnames(x_train_entry) %in% c(name)]
        x_test_entry = x_test_entry[, !colnames(x_test_entry) %in% c(name)]
      }
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
    
    ## imputing data
    
    # get indices with incomplete outcome
    incomplete_outcome_train_ind = which(is.na(y_train_entry)==T)
    
    # define data to impute
    # use outcome in imputation 
    if (include_outcome_imputation==T){
      # include test data in dataset to impute, but do not use it to calculate the imputations!
      if (include_missing_outcome_test==T|include_missing_predictors_test==T){
        rows_to_ignore = c(rep(FALSE,nrow(x_train_entry)),rep(TRUE,nrow(x_test_entry)))
        data_to_impute = rbind(cbind(x_train_entry,y_train_entry),cbind(x_test_entry,y_test_entry))
      } else { # don't include test data
        rows_to_ignore = c(rep(FALSE,nrow(x_train_entry)))
        data_to_impute = cbind(x_train_entry,y_train_entry)
      }
    } else { # don't use outcome in imputation 
      # include test data in dataset to impute, but do not use it to calculate the imputations!
      if (include_missing_outcome_test==T|include_missing_predictors_test==T){
        rows_to_ignore = c(rep(FALSE,nrow(x_train_entry)),rep(TRUE,nrow(x_test_entry)))
        data_to_impute = rbind(x_train_entry,x_test_entry)
      } else { # don't include test data
        rows_to_ignore = c(rep(FALSE,nrow(x_train_entry)))
        data_to_impute = x_train_entry
        } 
    }
    
    # imputation
    invisible(capture.output({
      data_imputed = mice(data_to_impute,m=m_mice,method=method_mice,seed=seed,ignore = rows_to_ignore)
    }))
    
    # get indices to include in the imputed train set
    if (MID==T){ # if 'Multiple imputation then delete (MID)' is true
      train_indices_imputed = c(1:nrow(x_train_entry))[c(1:nrow(x_train_entry))%!in%incomplete_outcome_train_ind]
    } else { # 'Multiple imputation then delete (MID)' is false
      train_indices_imputed = c(1:nrow(x_train_entry))
    }
  
    # find best elastic net model for each imputed data set
    results_imputed_all = foreach(imputation = 1:m_mice,.combine = 'rbind',.packages = c('mice','glmnet','ensr')) %dopar% {

      # get x_train an y_train imputed
      if (include_outcome_imputation==T){ # if the outcome was used in the imputation
          y_train_imputation = complete(data_imputed,imputation)[train_indices_imputed,ncol(complete(data_imputed,imputation))]
          x_train_imputation = complete(data_imputed,imputation)[train_indices_imputed,-ncol(complete(data_imputed,imputation))]
        }  else { # if the outcome was not used in the imputation
          y_train_imputation = y_train_entry[train_indices_imputed]
          x_train_imputation = complete(data_imputed,imputation)[train_indices_imputed,]
        }

      ## finding best lambda and alpha
      
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
      
      # store results
      results_imputation = data.frame(matrix(ncol=(ncol(x_train_entry)+1),nrow=1))
      colnames(results_imputation) = c('intercept',colnames(x_train_entry))
      results_imputation[1,1] = elastic_model$a0
      results_imputation[1,2:ncol(results_imputation)] = estimates
      
      return(results_imputation)
    }

    # calculate metrics
    
    # get a list of x test data if it was imputed
    if (include_missing_predictors_test==T|include_missing_outcome_test==T){
      x_test_entry = lapply(1:m_mice,function (x){
        complete(data_imputed,x)[-c(1:nrow(x_train_entry)),1:ncol(x_test_entry)]
      })
    } else { # create a alist if x test was not imputed
      x_test_entry = list(x_test_entry)
    }
  
    # aggregating imputed results
    results_imputed_aggregated = colMeans(results_imputed_all)
    
    # Stopping if there aren't enough observations in the training data
    if (is.null(stop_test)==F){
      if (sum(as.numeric(as.character(unlist(y_test_entry))),na.rm=T)<stop_test){next}
    }
    
    if (family=='binary'){
      # AUC, sensitivity, specificity
      predictions_list = lapply(x_test_entry,function (x){
        data.matrix(x)%*%results_imputed_aggregated[2:length(results_imputed_aggregated)]+results_imputed_aggregated[1]
      })
      predictions = as.numeric(rowMeans(do.call(cbind, predictions_list)))
      predictions = exp(predictions)/(1+exp(predictions))
      model_roc =  roc(unlist(y_test_entry),as.numeric(predictions),direction="<",quiet=T)
      model_coords = coords(model_roc,"best", ret=c("threshold", "specificity", "sensitivity"), transpose=FALSE)
      model_auc = auc(model_roc)
      ifelse( nrow(model_coords[2])>1,model_spec <- NA, model_spec <- model_coords[2])
      ifelse( nrow(model_coords[2])>1,model_sens <- NA, model_sens <- model_coords[3])
      
      # store predictions
      predictions_all[test_indices_all[[entry]]] = predictions
      
      # accuracy, PPV, NPV
      predictions_bin = ifelse(predictions>model_coords$threshold,1,0)
      confmatrix <- confusionMatrix(as.factor(predictions_bin),as.factor(unlist(y_test_entry)),positive='1')
      
      # storing metrics
      results_df[entry,'fold']=entry
      results_df[entry,'nrow_train']=nrow(x_train_entry)
      results_df[entry,'nrow_test']=nrow(x_test_entry[[1]])
      results_df[entry,'ny_train']=sum(as.numeric(as.character(unlist(y_train_entry))))
      results_df[entry,'ny_test']=sum(as.numeric(as.character(unlist(y_test_entry))))
      results_df[entry,'AUC']=model_auc
      results_df[entry,'sensitivity']=model_sens
      results_df[entry,'specificity']=model_spec
      results_df[entry,'accuracy']=confmatrix$overall[1]
      results_df[entry,'PPV']=confmatrix$byClass[3]
      results_df[entry,'NPV']=confmatrix$byClass[4]
      
    }
    else if (family=='continuous'){
      
      # Getting the predictions
      predictions_list = lapply(x_test_entry,function (x){
        data.matrix(x)%*%results_imputed_aggregated[2:length(results_imputed_aggregated)]+results_imputed_aggregated[1]
      })
      predictions = as.numeric(rowMeans(do.call(cbind, predictions_list)))
      if (is.null(pred_min)==F){predictions[predictions<pred_min]=pred_min}
      if (is.null(pred_max)==F){predictions[predictions>pred_max]=pred_max}
      
      # store predictions
      predictions_all[test_indices_all[[entry]]] = predictions
      
      # Getting R2, adjusted R2, RMSE, MSE, MAE
      R2 = 1-(sum((y_test_entry-predictions)^2)/length(unlist(y_test_entry)))/(sum((y_test_entry-mean(unlist(y_train_entry)))^2)/length(unlist(y_test_entry)))
      R2_adjusted = 1 - (1-R2)*(length(unlist(y_test_entry))-1)/(length(unlist(y_test_entry))-length(which(estimates!=0))-1)
      RMSE = RMSE(unlist(y_test_entry),predictions)
      MSE = RMSE^2
      MAE = MAE(unlist(y_test_entry),predictions)
      
      # storing metrics
      results_df[entry,'fold']=entry
      results_df[entry,'nrow_train']=nrow(x_train_entry)
      results_df[entry,'nrow_test']=nrow(x_test_entry[[1]])
      results_df[entry,'mean_y_train']=mean(unlist(y_train_entry))
      results_df[entry,'mean_y_test']=mean(unlist(y_test_entry))
      results_df[entry,'R2']=R2
      results_df[entry,'R2_adjusted']=R2_adjusted
      results_df[entry,'RMSE']=RMSE
      results_df[entry,'MSE']=MSE
      results_df[entry,'MAE']=MAE
    }
    
    # storing estimates
    for (predictor in predictors){
      index = which(names(results_imputed_aggregated)==predictor)
      if (length(index)==0){
        results_df[entry,predictor]<- NA
      }
      else{
        results_df[entry,predictor]<- results_imputed_aggregated[index]
      }
    }
    
    # updating progress bar
    setTxtProgressBar(pb,entry)
  }
  
  # close progress bar
  close(pb)
  
  # remove stored variables
  rm(x_train_entry,envir = .GlobalEnv)
  rm(y_train_entry,envir = .GlobalEnv)
  rm(ensr_lambdas,envir = .GlobalEnv)
  rm(ensr_cv,envir = .GlobalEnv)
  rm(ensr_alphas,envir = .GlobalEnv)
  
  # Create final results
  results = list()
  results$metrics = results_df
  results$predictions = predictions_all
  
  # stop cluster
  stopCluster(cl = cluster)
  
  # return results
  return(results)
}
