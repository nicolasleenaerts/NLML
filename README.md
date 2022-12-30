# MBR_ML

This github repository stores the machine learning tools of the Mind Body Research Group of the KU Leuven. It includes several scripts using elastic net regularized regression, BISCUIT, random forest and gradient boosting. The scripts in this repository were developed to build person-specific and pooled prediction models for binge eating, binge drinking and alcohol use with ecological momentary assessment data. However, they can be used in all kinds of contexts.

### Elastic Net

Elastic net regularized regression can be a powerful machine learning technique that reduces overfitting and can work with high-dimensional data. Furthermore, it can also provide information on the strength and nature of the relation between different predictors and the outcome. It does so by combining two regularization methods, ridge regression (L2) which shrinks model estimates and lasso regression (L1) which removes variables that don’t contribute to the model. The amount of ridge and lasso regression is expressed by a variable alpha which varies from 0 (exclusively ridge regression) to 1 (exclusively LASSO). The strength of the regularization is defined by a variable lambda with higher values of lambda leading to more shrinkage of the coefficients. The best alpha and lambda are selected through a grid search. 

## Tutorial

* Train Test Split or k-Fold Cross-Validation

The wrappers give you the possibility to perform a train test split. The default setting of the wrappers is to use 80% of the data as training data and to use 20% as testing data. However, you can change this by entering the percentage of training data you want with the 'split' argument. For example, if you want to use 50% of the data as training data, then you enter split=50.

`elastic_net_wrapper(data=my_data,outcome='binge_eating',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,split =50)`

The wrappers also give you the possibility to perform a k-fold cross validation. If you wan to do so, you have to specify the number of folds you want with the 'outer_cv' argument. For example, if you want to perform a 5-fold cross-validation, then you enter outer_cv=5.

`elastic_net_wrapper(data=my_data,outcome='binge_eating',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,outer_cv=5)`

* Stratification

By default, the train test split or k-fold cross-validation is stratified. This means that the distribution of the outcome is the same across splits or folds. If you don't want the split or cross-validation to be stratified, then you enter stratified=F.

`elastic_net_wrapper(data=my_data,outcome='binge_eating',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,stratified=F)`
