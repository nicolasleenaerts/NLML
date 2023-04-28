# MBR_ML

This github repository stores the machine learning tools of the Mind Body Research Group of the KU Leuven. The scripts in this repository were developed to build person-specific and pooled prediction models for binge eating, binge drinking and alcohol use with ecological momentary assessment data. However, they can be used in all kinds of contexts.

### Elastic Net

Elastic net regularized regression can be a powerful machine learning technique that reduces overfitting and can work with high-dimensional data. Furthermore, it can also provide information on the strength and nature of the relation between different predictors and the outcome. It does so by combining two regularization methods, ridge regression (L2) which shrinks model estimates and LASSO regression (L1) which removes variables that don’t contribute to the model. The amount of ridge and lasso regression is expressed by a variable alpha which varies from 0 (exclusively ridge regression) to 1 (exclusively LASSO). The strength of the regularization is defined by a variable lambda with higher values of lambda leading to more shrinkage of the coefficients. The best alpha and lambda are selected through a grid search. 

## Arguments

* data

Dataframe which includes the outcome and the predictor variables

* by

Only included in the pooled wrapper. A list of elements by which the data is grouped. 

* predictors_con

A list of continuous predictors that will be used to predict the outcome. In the pooled wrapper, the within-subject continuous predictors have to be entered here.

* predictors_cat

A list of categorical predictors that will be used to predict the outcome. In the pooled wrapper, the within-subject categorical predictors have to be entered here.

* between_predictors_con

A list of between-subject continuous predictors that will be used to predict the outcome. Only available in the pooled wrapper.

* between_predictors_cat

A list of between-subject categorical predictors that will be used to predict the outcome. Only available in the pooled wrapper.

* split

A number indicating the percentage of the data that will be used as training data. The rest of the data is used as test data. This argument is ignored when nested cross-validation is used. The default value is 80.

* outer_cv

A number defining how many folds will be used in the outer cross-validation loop. Specifying a number here will make the wrapper use nested cross-validation.

* stratified

A logical indicating whether the train test split or cross-validation needs to be stratified. The default is TRUE.

* scaling

A logical defining whether the continuous predictors need to be scaled. In the pooled wrapper, the within-subject continuous predictors are scaled at the within-person level and the between-subject continuous predictors are scaled at the between-subject level. The default is TRUE.

* repeated_cv 

A number indicating how many times the inner cross-validation loop needs to be repeated. The default value is 1, meaning that the inner cross-validation loop is not repeated.

* ensr_cv

A number defining how many folds will be used in the inner cross-validation loop. The default value is 10. 

* ensr_alphas

A list of numbers ranging between 0 and 1 that will be used in in the grid search to find the best alpha and lambda. The default values are 0.0000000 0.1111111, 0.2222222, 0.3333333, 0.4444444, 0.5555556, 0.6666667, 0.7777778, 0.8888889, 1.0000000.

* ensr_lambdas

A number identifying how many lambda values will be used in the grid search to find the best alpha and lambda. The default value is 100.

* stop_train

A number specifying the minimum number of observations the train data needs to have before the model is trained.

* stop_test

A number defining the minimum number of observations the test data needs to have before the model is evaluated.

* family

A character specifying whether the outcome is 'binary' or 'contiunous'. The default is 'binary'.

* shuffle

A logical identifying whether the data need to be shuffled before the cross-validation

* seed

A number defining the seed which is used for the steps of the wrapper that are random.

* pred_min

A number defining the lower end of the predictions for continuous outcomes.

* pred_max

A number specifying the upper end of the predictions for continuous outcomes.




* Train Test Split or k-Fold Cross-Validation

The wrappers give you the possibility to perform a train test split. The default setting of the wrappers is to use 80% of the data as training data and to use 20% as testing data. However, you can change this by entering the percentage of training data you want with the 'split' argument. For example, if you want to use 50% of the data as training data, you enter split=50.

`elastic_net_wrapper(data=my_data,outcome='binge_eating',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,split=50)`

The wrappers also give you the possibility to perform a k-fold cross validation. If you want to do so, you have to specify the number of folds you want with the 'outer_cv' argument. For example, if you want to perform a 5-fold cross-validation, then you enter outer_cv=5.

`elastic_net_wrapper(data=my_data,outcome='binge_eating',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,outer_cv=5)`

* Stratification

By default, the train test split or k-fold cross-validation is stratified. This means that the distribution of the outcome is the same across splits or folds. If you don't want the split or cross-validation to be stratified, then you enter stratified=F.

`elastic_net_wrapper(data=my_data,outcome='binge_eating',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,stratified=F)`

* Outcome

The wrappers can handle binary as well as continuous outcomes. You have to specifify the outcome variable with the 'outcome' argument and then specifify its nature with the 'family' argument (i.e., 'binary' or 'continuous'). If you have a binary outcome, then the wrapper wil calculate the area under the curve (AUD), sensitivity, specifivity, positive predictive value (PPV), and negative predictive value (NPV). If you have a continuous outcome, the wrapper will calculate the R2, the root mean squared error (RMSE), mean squared error (MSE), and mean absolute error (MAE)

* Predictions

You have the option to specify the minumum and maximum predicted outcome for continuous outcomes with the 'pred_min' and 'pred_max' arguments. That way, you can avoid that the elastic net algorithms makes impossible predictions. For example, if your outcome varies between 0 en 100, you can enter pred_min=0 and pred_max=100

`elastic_net_wrapper(data=my_data,outcome='loss_of_control',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,
family='continuous',pred_min=0,pred_max=100)`

* Predictors 

You have to enter continuous and categorical predictors seperately with the 'predictors_con' and 'predictors_cat' arguments. Furthermore, if you are using the wrapper for the pooled models, then you have to seperate within- and between-subject predictors by entering the within-subject predictors with the 'predictors_con' and 'predictors_cat' arguments and the between-subject predictors with the 'between_predictors_con' and 'between_predictors_cat' arguments. 

`elastic_net_wrapper_pooled(data=my_data,outcome='loss_of_control',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,
between_predictors_con = between_predictors_con, between_predictors_cat = between_predictors_cat)`

* Scaling
The wrapper can standardize the continuous predictors with the 'scale' argument. The pooled wrapper will scale the within-subject predictors on the within-subject on the within-subject level and the between-subject predictors on the between-subject level. You can ask the wrapper to scale the data with by entering scale=T.

`elastic_net_wrapper_pooled(data=my_data,outcome='loss_of_control',predictors_con=predictor_variables_con,predictors_cat=predictor_variables_cat,
between_predictors_con = between_predictors_con, between_predictors_cat = between_predictors_cat, scale =T)`



