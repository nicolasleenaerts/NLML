# MBR_ML

This github repository stores the machine learning tools of the Mind Body Research Group of the KU Leuven. The scripts in this repository were developed to build person-specific and pooled prediction models for binge eating, binge drinking and alcohol use with ecological momentary assessment data. However, they can be used in all kinds of contexts.

### Elastic Net

Elastic net regularized regression can be a powerful machine learning technique that reduces overfitting and can work with high-dimensional data. Furthermore, it can also provide information on the strength and nature of the relation between different predictors and the outcome. It does so by combining two regularization methods, ridge regression (L2) which shrinks model estimates and lasso regression (L1) which removes variables that don’t contribute to the model. The amount of ridge and lasso regression is expressed by a variable alpha which varies from 0 (exclusively ridge regression) to 1 (exclusively LASSO). The strength of the regularization is defined by a variable lambda with higher values of lambda leading to more shrinkage of the coefficients. The best alpha and lambda are selected through a grid search. 

## Explanation

* Train Test Split or k-Fold Cross-Validation

The wrappers give you the possibility to perform a train test split. The default setting of the wrappers is to use 80% of the data as training data and to use 20% as testing data. However, you can change this by entering the percentage of training data you want with the 'split' argument. For example, if you want to use 50% of the data as training data, then you enter split=50.

`elastic_net_wrapper(data=my_data,outcome='binge_eating',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,split =50)`

The wrappers also give you the possibility to perform a k-fold cross validation. If you wan to do so, you have to specify the number of folds you want with the 'outer_cv' argument. For example, if you want to perform a 5-fold cross-validation, then you enter outer_cv=5.

`elastic_net_wrapper(data=my_data,outcome='binge_eating',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,outer_cv=5)`

* Stratification

By default, the train test split or k-fold cross-validation is stratified. This means that the distribution of the outcome is the same across splits or folds. If you don't want the split or cross-validation to be stratified, then you enter stratified=F.

`elastic_net_wrapper(data=my_data,outcome='binge_eating',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,stratified=F)`

* Outcome

The wrappers can handle binary as well as continuous outcomes. You have to specifify the outcome variable with the 'outcome' argument and then specifify its nature with the 'family' argument (i.e., 'binary' or 'continuous'). If you have a binary outcome, then the wrapper wil calculate the area under the curve (AUD), sensitivity, specifivity, positive predictive value (PPV), and negative predictive value (NPV).

You have the option of specifying the minumum and maximum predicted outcome for continuous outcomes with the 'pred_min' and 'pred_max' arguments. That way, you can avoid that the elastic net algorithms makes impossible predictions.

* Predictors 

You have to enter continuous and categorical predictors seperately with the 'predictors_con' and 'predictors_cat' arguments. Furthermore, if you are using the wrapper for the pooled models, then you have to seperate within- and between-subject predictors by entering the within-subject predictors with the 'predictors_con' and 'predictors_cat' arguments and the between-subject predictors with the 'between_predictors_con' and 'between_predictors_cat' arguments. 

The wrapper can stadardize the continuous predictors with the 'scale' argument. The pooled wrapper while scale the predictors specifief 



