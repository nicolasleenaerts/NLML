# MBR_ML

This github repository stores the machine learning tools of the Mind Body Research Group of the KU Leuven. It includes several scripts using elastic net regularized regression, BISCUIT, random forest and gradient boosting. The scripts in this repository were developed to build person-specific and pooled prediction models for binge eating, binge drinking and alcohol use with ecological momentary assessment data. However, they can be used in all kinds of contexts.

### Elastic Net

Elastic net regularized regression can be a powerful machine learning technique that reduces overfitting and can work with high-dimensional data. Furthermore, it can also provide information on the strength and nature of the relation between different predictors and the outcome. It does so by combining two regularization methods, ridge regression (L2) which shrinks model estimates and lasso regression (L1) which removes variables that don’t contribute to the model. The amount of ridge and lasso regression is expressed by a variable alpha which varies from 0 (exclusively ridge regression) to 1 (exclusively LASSO). The strength of the regularization is defined by a variable lambda with higher values of lambda leading to more shrinkage of the coefficients. The best alpha and lambda are selected through a grid search. 

### Biscuit

### Random Forest

### Gradient Boosting
