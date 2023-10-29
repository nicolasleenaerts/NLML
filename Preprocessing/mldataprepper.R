mldataprepper <- function(data, outcome=NULL, predictors_con=NULL,predictors_cat=NULL, split=80, outer_cv=NULL, stratified=T,scaling=T,
                                seed=404,shuffle=T,clean_columns=T){
  # required packages
  require(splitTools)
  `%!in%` = Negate(`%in%`)
  
  # combine predictors
  predictors = c(predictors_con,predictors_cat)
  
  # shuffle dataset to lose time contingency for CV
  if (shuffle==T){
    set.seed(seed)
    data = data[sample(nrow(data)),]
  }
  
  # split data into y and x
  y = data[outcome]
  x = data[predictors]
  
  # create list of x and y datasets to be analyzed
  analysis_list = list()
  # split x and y into training and testing data
  if (is.null(outer_cv)==T){
    # performing stratified split
    if(stratified==T){
      if (family=='binary'){
        # get indices
        set.seed(seed)
        my_train_ind_no_y =  sample(which(y==0), size = split/100*length(which(y==0)))
        set.seed(seed)
        my_train_ind_y =  sample(which(y==1), size = split/100*length(which(y==1)))
        # split data
        y_train = y[c(my_train_ind_no_y,my_train_ind_y),]
        y_test =y[-c(my_train_ind_no_y,my_train_ind_y),]
        x_train = x[c(my_train_ind_no_y,my_train_ind_y),]
        x_test = x[-c(my_train_ind_no_y,my_train_ind_y),]
        # add to analysis list
        analysis_list[[1]] = list(y_train,y_test,x_train,x_test)
      }
      else if (family=='continuous'){
        # get indices
        set.seed(seed)
        my_train_ind =  createDataPartition(as.matrix(y), p = split/100, list = T,groups=min(3,nrow(y)))
        # split data
        y_train = y[c(unlist(my_train_ind)),]
        y_test =y[-c(unlist(my_train_ind)),]
        x_train = x[c(unlist(my_train_ind)),]
        x_test = x[-c(unlist(my_train_ind)),]
        # add to analysis list
        analysis_list[[1]] = list(y_train,y_test,x_train,x_test)
      }
    }
    # performing non-stratified split
    else if(stratified==F){
      set.seed(seed)
      my_train_ind =  sample(c(1:nrow(y)), size = split/100*nrow(y))
      y_train = y[c(my_train_ind),]
      y_test =y[-c(my_train_ind),]
      x_train = x[c(my_train_ind),]
      x_test = x[-c(my_train_ind),]
      # add to analysis list
      analysis_list[[1]] = list(y_train,y_test,x_train,x_test)
    }
  }
  # creating datasets for cross-validation
  else {
    if(stratified==T){
      # creating folds
      set.seed(seed)
      folds <- create_folds(as.numeric(unlist(y)),k = outer_cv,type='stratified')
      # creating datasets
      for(nfold in 1:length(folds)){
        y_train <- y[c(folds[[nfold]]), ]
        y_test <- y[-c(folds[[nfold]]), ]
        x_train <- x[c(folds[[nfold]]), ]
        x_test <- x[-c(folds[[nfold]]), ]
        analysis_list[[nfold]] = list(y_train,y_test,x_train,x_test)
      }
    }
    else if(stratified==F){
      set.seed(seed)
      folds <- create_folds(as.numeric(unlist(y)),k = outer_cv,type='basic')
      for(nfold in 1:length(folds)){
        y_train <- y[c(folds[[nfold]]), ]
        y_test <- y[-c(folds[[nfold]]), ]
        x_train <- x[c(folds[[nfold]]), ]
        x_test <- x[-c(folds[[nfold]]), ]
        analysis_list[[nfold]] = list(y_train,y_test,x_train,x_test)
      }
    }
  }
  
  # Training and testing the elastic net
  for (entry in 1:length(analysis_list)){
    
      #scaling numeric data
      if (scaling==T){
        for(variable in predictors_con){
          mean_variable = mean(as.numeric(unlist(analysis_list[[entry]][3][,variable])),na.rm=T)
          sd_variable = sd(as.numeric(unlist(analysis_list[[entry]][3][,variable])),na.rm=T)
          analysis_list[[entry]][3][,variable] = (as.numeric(unlist(analysis_list[[entry]][3][,variable]))-mean_variable)/sd_variable
          analysis_list[[entry]][4][,variable] = (as.numeric(unlist(analysis_list[[entry]][4][,variable]))-mean_variable)/sd_variable
        }
      }
      
      # removing variables with no variance from the training data
      if (clean_columns==T){
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