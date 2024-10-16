mldataprepper_pooled <- function(data, outcome=NULL, by=NULL,predictors_con=NULL,predictors_cat=NULL,
                                 between_predictors_con=NULL,between_predictors_cat=NULL,
                                 split=80, outer_cv=NULL,stratified=T,scaling=T,seed=404,clean_columns=T){
  # required packages
  require(caret)
  require(splitTools)
  `%!in%` = Negate(`%in%`)
  
  # combine predictors
  predictors = c(predictors_con,predictors_cat,between_predictors_con,between_predictors_cat)
  
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
        if (family=='binary'){
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
        }
        else if (family=='continuous'){
          # get indices
          set.seed(seed)
          my_train_ind =  createDataPartition(as.matrix(y_by), p = split/100, list = T,groups=min(3,nrow(y_by)))
          # split data
          y_train_by = y_by[c(unlist(my_train_ind)),]
          y_test_by =y_by[-c(unlist(my_train_ind)),]
          x_train_by = x_by[c(unlist(my_train_ind)),]
          x_test_by = x_by[-c(unlist(my_train_ind)),]
        }
        #scaling numeric data on the within level
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
        
        #scaling numeric data on the within level
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
          
          #scaling numeric data on the within level
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
          
          #scaling numeric data on the within level
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
  
  # Scaling numeric data on the between level
  if (scaling==T){
    for (entry in 1:length(analysis_list)){
      for(variable in between_predictors_con){
        mean_variable = mean(as.numeric(unlist(analysis_list[[entry]][[2]][,variable])),na.rm=T)
        sd_variable = sd(as.numeric(unlist(analysis_list[[entry]][[2]][,variable])),na.rm=T)
        if(sd_variable==0){
          analysis_list[[entry]][[2]][,variable] = as.numeric(unlist(analysis_list[[entry]][[2]][,variable]))-mean_variable
          analysis_list[[entry]][[4]] = lapply(analysis_list[[entry]][[4]], function (x){x[,variable]=(x[,variable]-mean_variable)/sd_variable 
          return(x)})
          next}
        analysis_list[[entry]][[2]][,variable] = (as.numeric(unlist(analysis_list[[entry]][[2]][,variable]))-mean_variable)/sd_variable
        analysis_list[[entry]][[4]] = lapply(analysis_list[[entry]][[4]], function (x){x[,variable]=(x[,variable]-mean_variable)/sd_variable 
        return(x)})
      }
    }
  }
  
  # removing variables with no variance from the training data
  if (clean_columns==T){
    for (entry in 1:length(analysis_list)){
      for (name in colnames(analysis_list[[entry]][3])){
        if (length(unique(analysis_list[[entry]][3][,name]))<2){
          analysis_list[[entry]][3] = analysis_list[[entry]][3][, !colnames(analysis_list[[entry]][3]) %in% c(name)]
          analysis_list[[entry]][4] = analysis_list[[entry]][4][, !colnames(analysis_list[[entry]][4]) %in% c(name)]
        }
      }
    }
  }
  
  # return df
  return(analysis_list)
}
