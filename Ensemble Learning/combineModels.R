combineModels <- function(models,outcome,predictors,learning_rate = 0.01, threshold=0.001,maxiter=1500,thetas=NULL,bias=NULL,method='combination') {
  
  # Load libraries
  require(dplyr)
  
  # Create prediction matrix
  prediction_matrix=data.matrix(data.frame(lapply(models,function (x) predict(x,data.matrix(select(predictors,rownames(x$beta))),type='response'))))
  
  # Initialize theta's and bias
  if (is.null(thetas)==T){thetas = rep(0,ncol(prediction_matrix))}
  if (is.null(bias)==T){bias = 0}
  
  # Find optimal theta's
  for (iteration in 1:maxiter){
    
    if (method=='combination'){
      # Calculating the gradients for the weights with the loss function
      gradients_theta = (1 / length(outcome)) * (t(prediction_matrix) %*% ((prediction_matrix %*% thetas) - outcome))
      
      # Adjust thetas
      thetas = thetas-learning_rate*gradients_theta
    }
    
    else if (method=='logistic'){
      # Calculating the gradients for the weights with the loss function
      gradients_theta = (1 / length(outcome)) * (t(prediction_matrix) %*% (1/(1 + exp(-(prediction_matrix %*% thetas+bias))) - outcome))
      
      # Calculating the gradient for the bias with the loss function
      gradient_bias = (1 / length(outcome)) * sum(1/(1 + exp(-(prediction_matrix %*% thetas+bias))) - outcome)
      
      # Adjust thetas
      thetas = thetas-learning_rate*gradients_theta
      
      # Adjust bias
      bias = bias-learning_rate*gradient_bias
    }
    # Check tolerance
    if(all(abs(gradients_theta*learning_rate)<threshold)){break}
  }
  
  # Create results list
  results_list = list()
  
  # Store the results
  results_list$thetas = thetas
  results_list$bias = bias
  
  # Store convergence
  if (iteration < maxiter){results_list$converged='True'}
  else {results_list$converged='False'}
  
  # Return results
  return(results_list)
}
