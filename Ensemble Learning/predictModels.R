predictModels <- function(combineModels,models,newdata,method='combination'){
  
  # Load libraries
  require(dplyr)
  
  # Create prediction matrix
  prediction_matrix=data.matrix(data.frame(lapply(models,function (x) predict(x,data.matrix(select(newdata,rownames(x$beta))),type='response'))))
  
  # Predict
  if (method=='combination'){
    predictions=as.matrix(prediction_matrix) %*% combineModels$thetas
  }
  
  else if (method=='logistic'){
    predictions=1/(1 + exp(-(as.matrix(prediction_matrix) %*% combineModels$thetas+combineModels$bias)))
  }
  
  # Return results
  return(predictions)
}