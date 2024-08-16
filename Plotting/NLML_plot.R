NLML_plot <- function(results_estimates,percentile=0.90,range=TRUE,title=NULL,subtitle=NULL,xlim=NULL,ylab='Predictor',xlab='Estimate', gradient_values= c(-1.5,-1,-0.5,0,0.25,0.5,0.75,1,1.5,2,2.5)){
  # Loading libraries
  require(ggplot2)
  require(dplyr)
  require(tidyr)
  require(forcats)
  require(scales)
  require(colorspace)
  
  # Summarize results
  results_estimates_long <- gather(results_estimates, variable, estimate, colnames(results_estimates)[1]:colnames(results_estimates)[ncol(results_estimates)], factor_key=TRUE)
  summary_estimates <- results_estimates_long %>%
    group_by(variable) %>%
    summarise(mean_estimate = mean(estimate,na.rm=T),P025 = quantile(estimate, 0.025,na.rm=TRUE),P975 = quantile(estimate, 0.975,na.rm=TRUE))
  
  # Set colors
  green_red_palette <- brewer_pal(type = "div", palette = "RdYlGn")(11)
  green_red_palette_dark <- darken(green_red_palette, amount = 0.3)
  green_red_palette_dark_less <- darken(green_red_palette, amount = 0.075)
  
  # Set limits 
  if (is.null(xlim)==T){
    xlim = c(min(summary_estimates$P025),max(summary_estimates$P975))
  }
  
  # Make base plot
  base_plot <- ggplot(summary_estimates[abs(summary_estimates$mean_estimate)>=
                                          quantile(abs(summary_estimates$mean_estimate),percentile,na.rm=TRUE),],
                      aes(x=mean_estimate,y=fct_reorder(variable,mean_estimate)))
  
  # Add 95% segments
  if (range==TRUE){
    base_plot <- base_plot + geom_segment(aes(x=P025,xend=P975,yend=variable,color=mean_estimate),size=3)
  }
  
  # Finish plot
  plot <- base_plot +
    
    # Add 0 line
    geom_vline(xintercept = 0, lty = 2, size = 0.2) +
    
    # Add means
    geom_point(aes(fill = mean_estimate),pch = 21,size=3) +
    
    # Change colors
    scale_fill_gradientn(colours = green_red_palette_dark,values = gradient_values)+
    scale_color_gradientn(colours = green_red_palette_dark_less,values = gradient_values)+
    
    # Remove background
    theme_bw()+
    
    # Set names of the x and y axis
    xlab(xlab)+
    ylab(ylab)+
    
    # Set the x-axis at the top
    scale_x_continuous(position = "top")+
    
    # Change themes
    theme(legend.position="none",
          axis.text.y = element_text(size = 8,face = "bold"),
          axis.ticks.y = element_blank(),
          axis.line = element_blank(),
          plot.title = element_text(hjust = 0.5,face = "bold"),
          plot.subtitle = element_text(hjust = 0.5,face = "bold"))+
    
    # Correct zoom x
    coord_cartesian(xlim=xlim)+
    
    # Title
    ggtitle(title,subtitle = subtitle)
  
  # Return plot
  return(plot)
}
