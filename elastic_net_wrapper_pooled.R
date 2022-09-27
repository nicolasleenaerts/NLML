elastic_net_wrapper_pooled <- function(data, outcome=NULL, by=NULL,predictors_con=NULL,predictors_cat=NULL, split=80, outer_cv=NULL, 
                                       stratified=T,scaling=T,repeated_cv=1,ensr_cv=10,ensr_alphas=seq(0, 1, length = 10),ensr_lambdas=100,seed=404,
                                       stop_test=NULL,shuffle=T){
  # required packages
  require(ensr)
  require(glmnet)
  require(pROC)
  require(caret)
  require(splitTools)
  `%!in%` = Negate(`%in%`)
  
  # combine predictors
  predictors = c(predictors_con,predictors_cat)
  
  # create list of x and y datasets to be analyzed
  analysis_list = list()
  if (is.null(outer_cv)==T){
    analysis_list[[1]] = list()
    analysis_list[[1]][[1]] = data.frame()
    analysis_list[[1]][[2]] = data.frame()
    analysis_list[[1]][[3]] = list()
    analysis_list[[1]][[4]] = list()
  }
  else{
    for(nfold in 1:outer_cv){
      analysis_list[[nfold]] = list()
      analysis_list[[nfold]][[1]] = data.frame()
      analysis_list[[nfold]][[2]] = data.frame()
      analysis_list[[nfold]][[3]] = list()
      analysis_list[[nfold]][[4]] = list()
    }
  }
  
  # Loop over by
  for (by_index in 1:length(unique(unlist(data[by])))){
    by_entry = unique(unlist(data[by]))[by_index]
    data_by = subset(data,unlist(data[by])==by_entry)
    
    # shuffle dataset to lose time contingency for CV
    if (shuffle==T){
      set.seed(seed)
      data_by = data_by[sample(nrow(data_by)),]
    }
    
    # split data into y and x
    y_by = data_by[outcome]
    x_by = data_by[predictors]
    
    # split x and y into training and testing data
    if (is.null(outer_cv)==T){
      # performing stratified split
      if(stratified==T){
        # get indices
        set.seed(seed)
        my_train_ind_no_y =  sample(which(y_by==0), size = split/100*length(which(y_by==0)))
        set.seed(seed)
        my_train_ind_y =  sample(which(y_by==1), size = split/100*length(which(y_by==1)))
        # split data
        y_train_by = y_by[c(my_train_ind_no_y,my_train_ind_y),]
        y_test_by =y_by[-c(my_train_ind_no_y,my_train_ind_y),]
        x_train_by = x_by[c(my_train_ind_no_y,my_train_ind_y),]
        x_test_by = x_by[-c(my_train_ind_no_y,my_train_ind_y),]
        #scaling numeric data
        if (scaling==T){
          for(variable in predictors_con){
            mean_variable = mean(as.numeric(unlist(x_train_by[,variable])),na.rm=T)
            sd_variable = sd(as.numeric(unlist(x_train_by[,variable])),na.rm=T)
            if(sd_variable==0){
              x_train_by[,variable] = as.numeric(unlist(x_train_by[,variable]))-mean_variable
              x_test_by[,variable] = as.numeric(unlist(x_test_by[,variable]))-mean_variable
              next}
            x_train_by[,variable] = (as.numeric(unlist(x_train_by[,variable]))-mean_variable)/sd_variable
            x_test_by[,variable] = (as.numeric(unlist(x_test_by[,variable]))-mean_variable)/sd_variable
          }
        }   
        # add to analysis list
        analysis_list[[1]][[1]][(nrow(analysis_list[[1]][[1]])+1):(nrow(analysis_list[[1]][[1]])+length(y_train_by)),1] = y_train_by
        analysis_list[[1]][[2]][(nrow(analysis_list[[1]][[2]])+1):(nrow(analysis_list[[1]][[2]])+nrow(x_train_by)),1:ncol(x_train_by)] = x_train_by
        analysis_list[[1]][[3]][[by_index]] = y_test_by
        analysis_list[[1]][[4]][[by_index]] = x_test_by
      }
      # performing non-stratified split
      else if(stratified==F){
        set.seed(seed)
        my_train_ind =  sample(c(1:nrow(y_by)), size = split/100*nrow(y_by))
        y_train_by = y_by[c(my_train_ind),]
        y_test_by =y_by[-c(my_train_ind),]
        x_train_by = x_by[c(my_train_ind),]
        x_test_by = x_by[-c(my_train_ind),]
        
        #scaling numeric data
        if (scaling==T){
          for(variable in predictors_con){
            mean_variable = mean(as.numeric(unlist(x_train_by[,variable])),na.rm=T)
            sd_variable = sd(as.numeric(unlist(x_train_by[,variable])),na.rm=T)
            if(sd_variable==0){
              x_train_by[,variable] = as.numeric(unlist(x_train_by[,variable]))-mean_variable
              x_test_by[,variable] = as.numeric(unlist(x_test_by[,variable]))-mean_variable
              next}
            x_train_by[,variable] = (as.numeric(unlist(x_train_by[,variable]))-mean_variable)/sd_variable
            x_test_by[,variable] = (as.numeric(unlist(x_test_by[,variable]))-mean_variable)/sd_variable
          }
        }   
        
        # add to analysis list
        analysis_list[[1]][[1]][(nrow(analysis_list[[1]][[1]])+1):(nrow(analysis_list[[1]][[1]])+length(y_train_by)),1] = y_train_by
        analysis_list[[1]][[2]][(nrow(analysis_list[[1]][[2]])+1):(nrow(analysis_list[[1]][[2]])+nrow(x_train_by)),1:ncol(x_train_by)] = x_train_by
        analysis_list[[1]][[3]][[by_index]] = y_test_by
        analysis_list[[1]][[4]][[by_index]] = x_test_by
      }
    }
    # creating datasets for cross-validation
    else {
      if(stratified==T){
        # creating folds
        set.seed(seed)
        folds <- create_folds(as.numeric(unlist(y_by)),k = outer_cv,type='stratified')
        # creating datasets
        for(nfold in 1:length(folds)){
          y_train_by <- y_by[c(folds[[nfold]]), ]
          y_test_by <- y_by[-c(folds[[nfold]]), ]
          x_train_by <- x_by[c(folds[[nfold]]), ]
          x_test_by <- x_by[-c(folds[[nfold]]), ]
          
          #scaling numeric data
          if (scaling==T){
            for(variable in predictors_con){
              mean_variable = mean(as.numeric(unlist(x_train_by[,variable])),na.rm=T)
              sd_variable = sd(as.numeric(unlist(x_train_by[,variable])),na.rm=T)
              if(sd_variable==0){
                x_train_by[,variable] = as.numeric(unlist(x_train_by[,variable]))-mean_variable
                x_test_by[,variable] = as.numeric(unlist(x_test_by[,variable]))-mean_variable
                next}
              x_train_by[,variable] = (as.numeric(unlist(x_train_by[,variable]))-mean_variable)/sd_variable
              x_test_by[,variable] = (as.numeric(unlist(x_test_by[,variable]))-mean_variable)/sd_variable
            }
          }    
          
          # add to analysis list
          analysis_list[[nfold]][[1]][(nrow(analysis_list[[nfold]][[1]])+1):(nrow(analysis_list[[nfold]][[1]])+length(y_train_by)),1] = y_train_by
          analysis_list[[nfold]][[2]][(nrow(analysis_list[[nfold]][[2]])+1):(nrow(analysis_list[[nfold]][[2]])+nrow(x_train_by)),1:ncol(x_train_by)] = x_train_by
          analysis_list[[nfold]][[3]][[by_index]] = y_test_by
          analysis_list[[nfold]][[4]][[by_index]] = x_test_by
        }
      }
      else if(stratified==F){
        set.seed(seed)
        folds <- create_folds(as.numeric(unlist(y_by)),k = outer_cv,type='basic')
        for(nfold in 1:length(folds)){
          y_train_by <- y_by[c(folds[[nfold]]), ]
          y_test_by <- y_by[-c(folds[[nfold]]), ]
          x_train_by <- x_by[c(folds[[nfold]]), ]
          x_test_by <- x_by[-c(folds[[nfold]]), ]
          
          #scaling numeric data
          if (scaling==T){
            for(variable in predictors_con){
              mean_variable = mean(as.numeric(unlist(x_train_by[,variable])),na.rm=T)
              sd_variable = sd(as.numeric(unlist(x_train_by[,variable])),na.rm=T)
              if(sd_variable==0){
                x_train_by[,variable] = as.numeric(unlist(x_train_by[,variable]))-mean_variable
                x_test_by[,variable] = as.numeric(unlist(x_test_by[,variable]))-mean_variable
                next}
              x_train_by[,variable] = (as.numeric(unlist(x_train_by[,variable]))-mean_variable)/sd_variable
              x_test_by[,variable] = (as.numeric(unlist(x_test_by[,variable]))-mean_variable)/sd_variable
            }
          }      
          
          # add to analysis list
          analysis_list[[nfold]][[1]][(nrow(analysis_list[[nfold]][[1]])+1):(nrow(analysis_list[[nfold]][[1]])+length(y_train_by)),1] = y_train_by
          analysis_list[[nfold]][[2]][(nrow(analysis_list[[nfold]][[2]])+1):(nrow(analysis_list[[nfold]][[2]])+nrow(x_train_by)),1:ncol(x_train_by)] = x_train_by
          analysis_list[[nfold]][[3]][[by_index]] = y_test_by
          analysis_list[[nfold]][[4]][[by_index]] = x_test_by
        }
      }
    }
  }
  
  # creating the results dataframe
  results_list = list()
  results_df_model = data.frame(matrix(ncol = (3+length(predictors))))
  colnames(results_df_model) = c('fold','nrow_train','ny_train',predictors)
  results_df_pooled = data.frame(matrix(ncol = 10))
  colnames(results_df_pooled) = c('by','fold','nrow_test','ny_test','AUC','sensitivity','specificity',
                                  'accuracy','PPV','NPV')
  
  # Create progress bar
  print('Training and evaluating the models')
  pb = txtProgressBar(min = 0, max = length(analysis_list), initial = 0) 
  
  # Training and testing the elastic net
  for (entry in 1:length(analysis_list)){
    
    # getting the training and testing data
    y_train_entry = analysis_list[[entry]][[1]]
    x_train_entry= analysis_list[[entry]][[2]]
    
    # identify binary data
    binary_predictors = colnames(x_train_entry)[which(apply(x_train_entry,2,function(x) { all(x %in% 0:1) })==T)]
    binary_predictors = subset(binary_predictors,binary_predictors%!in%colnames(x_train_entry)[grepl('numeric',sapply(x_train_entry,class))])
    
    # transforming to a data matrix
    x_train_entry = data.matrix(x_train_entry)
    
    # correcting dummy coded variables
    x_train_entry[,c(binary_predictors)]<- x_train_entry[,c(binary_predictors)]-1
    
    # removing variables with no variance from the training data
    removed_names = c()
    for (name in colnames(x_train_entry)){
      if (length(unique(x_train_entry[,name]))<2){
        x_train_entry = x_train_entry[, !colnames(x_train_entry) %in% c(name)]
        removed_names[(length(removed_names)+1)]=name
      }
    }
    
    # finding best lambda and alpha
    
    # creating a variable for storing the crossvalidation results for the alphas and the lambdas
    MSEs <- NULL
    
    # store variables for  ensr
    x_train_entry <<- x_train_entry
    y_train_entry <<- y_train_entry
    ensr_lambdas <<- ensr_lambdas
    ensr_cv <<- ensr_cv
    ensr_alphas <<- ensr_alphas
    
    for (repeated_cv_number in 1:repeated_cv){
      
      # setting the seed
      set.seed(repeated_cv_number)
      # selecting the best alpha and lambda for this seed
      ensr_obj <- ensr(y =data.matrix(y_train_entry), x = x_train_entry,nlambda=ensr_lambdas,nfolds = ensr_cv,
                       alphas = ensr_alphas,family='binomial',standardize = F)
      ensr_obj_summary <- summary(object = ensr_obj)
      
      # storing the results
      MSEs <- cbind(MSEs,ensr_obj_summary$cvm)
    }
    
    # converting the cross validation results to a dataframe
    MSEs <- as.data.frame(MSEs)
    MSEs$rowMeans <- rowMeans(MSEs)
    
    # adding the alphas and lambdas that we used
    # these are the same for every seed!
    MSEs$lambdas <- ensr_obj_summary$lambda
    MSEs$alphas<- ensr_obj_summary$alpha
    MSEs <- MSEs[order(MSEs$rowMeans,decreasing = F), ]
    
    # Selecting the  alpha and the lambda of the best model
    alpha.min <- MSEs$alphas[1]
    lambda.min <- MSEs$lambdas[1]
    
    # fitting the elastic net model and getting the estimates for the variables
    elastic_model <- glmnet(y =data.matrix(y_train_entry), x = x_train_entry, family = 'binomial', alpha = alpha.min,
                            lambda=lambda.min,standardize = F)
    estimates <- elastic_model$beta
    
    # having at least one parameter
    while (length(which(estimates!=0))<1){
      MSEs <- MSEs[-1,]
      lambda.min <- MSEs$lambdas[1]
      alpha.min <- MSEs$alphas[1]
      elastic_model <- glmnet(y =data.matrix(y_train_entry), x = x_train_entry, family = 'binomial',
                              alpha = alpha.min,lambda=lambda.min,standardize = F)
      estimates <- elastic_model$beta
    }
    
    # Store predictors model
    results_df_model[entry,'fold'] = entry
    results_df_model[entry,'nrow_train']=nrow(x_train_entry)
    results_df_model[entry,'ny_train']=sum(as.numeric(as.character(unlist(y_train_entry))))
    
    # storing estimates
    for (predictor in predictors){
      index = which(rownames(estimates)==predictor)
      if (length(index)==0){
        results_df_model[entry,predictor]<- NA
      }
      else{
        results_df_model[entry,predictor]<- estimates[index]
      }
    }
    
    # calculate metrics model
    for (by_index in 1:length(unique(unlist(data[by])))){
      by_entry = unique(unlist(data[by]))[by_index]
      # getting test data
      y_test_by = analysis_list[[entry]][[3]][[by_index]]
      x_test_by = analysis_list[[entry]][[4]][[by_index]]
      
      # Stopping if there aren't enough observations in the training data
      if (is.null(stop_test)==F){
        if (sum(as.numeric(as.character(unlist(y_test_by))))<stop_test){next}
      }
      
      # having at least two levels
      if(length(unique(unlist(y_test_by)))<2){next}
      
      # transforming to a data matrix
      x_test_by = data.matrix(x_test_by)
      
      # correcting dummy coded variables
      x_test_by[,c(binary_predictors)]<- x_test_by[,c(binary_predictors)]-1
      
      # remove variables which were removed before
      for (name in removed_names){
        x_test_by = x_test_by[, !colnames(x_test_by) %in% c(name)]
      }
      
      # AUC, sensitivity, specificity
      predictions = predict(elastic_model, newx=x_test_by,type = "response")
      model_roc =  roc(unlist(y_test_by),as.numeric(predictions),direction="<",quiet=T)
      model_coords = coords(model_roc,"best", ret=c("threshold", "specificity", "sensitivity"), transpose=FALSE)
      model_auc = auc(model_roc)
      model_spec <- model_coords[2]
      model_sens <- model_coords[3]
      
      # accuracy, PPV, NPV
      predictions_bin = ifelse(predictions>model_coords$threshold,1,0)
      confmatrix <- confusionMatrix(as.factor(predictions_bin),as.factor(unlist(y_test_by)),positive='1')
      
      # storing metrics
      results_df_pooled[((entry-1)*length(unique(unlist(data[by])))+by_index),'by']=by_entry
      results_df_pooled[((entry-1)*length(unique(unlist(data[by])))+by_index),'fold']=entry
      results_df_pooled[((entry-1)*length(unique(unlist(data[by])))+by_index),'nrow_test']=nrow(x_test_by)
      results_df_pooled[((entry-1)*length(unique(unlist(data[by])))+by_index),'ny_test']=sum(as.numeric(as.character(unlist(y_test_by))))
      results_df_pooled[((entry-1)*length(unique(unlist(data[by])))+by_index),'AUC']=model_auc
      results_df_pooled[((entry-1)*length(unique(unlist(data[by])))+by_index),'sensitivity']=model_sens
      results_df_pooled[((entry-1)*length(unique(unlist(data[by])))+by_index),'specificity']=model_spec
      results_df_pooled[((entry-1)*length(unique(unlist(data[by])))+by_index),'accuracy']=confmatrix$overall[1]
      results_df_pooled[((entry-1)*length(unique(unlist(data[by])))+by_index),'PPV']=confmatrix$byClass[3]
      results_df_pooled[((entry-1)*length(unique(unlist(data[by])))+by_index),'NPV']=confmatrix$byClass[4]
    }
    
    # updating progress bar
    setTxtProgressBar(pb,entry)
    # remove stored variables
    rm(x_train_entry,envir = .GlobalEnv)
    rm(y_train_entry,envir = .GlobalEnv)
    rm(ensr_lambdas,envir = .GlobalEnv)
    rm(ensr_cv,envir = .GlobalEnv)
    rm(ensr_alphas,envir = .GlobalEnv)
  }
  # close progress bar
  close(pb)
  
  # Store results
  results_list$model_information=results_df_model
  results_list$model_results=results_df_pooled
  
  # Return results
  return(results_list)
}
