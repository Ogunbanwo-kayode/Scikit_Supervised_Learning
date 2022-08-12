# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 00:38:22 2017

@author: Shabaka
"""
#%%

    import numpy as np
    from random import randint
    import pandas as pd
    import seaborn as sns
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.metrics import roc_curve
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import Imputer
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import scale
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    

#%%
df = pd.read_csv('fixations.csv', index_col=0)

print(df.head())

#%%
################################################################
# ###################### Visual EDA ###############
percentile = np.percentile(df.confidence, range(0, 100, 25))

print('Percentile for confidence of data is ', percentile)

plt.figure()
_ = sns.lmplot(x='avg_pupil_size', y='duration', fit_reg=True,
               data=df, palette='RdBu')
plt.xticks([0, 1], ['High', 'Low'])
plt.show()

#%%
################################################################
# ################## Knn Classifier ################

# This cell will yield a label issue - rightly so too as the data being
# analysed is not fit for a classification type algorithm
# This is simply because the files I use only as placeholders here for guidance
# is dealing with a time series continuous data
# The structure is just a layout that works for the right sort of data - categoricals

# Create arrays for the features and the response variable
y = df['avg_pupil_size'].values
X = df.drop('avg_pupil_size', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

###############################################################
# ############## k-Nearest Neigbours using the predict() #####

# Create arrays for the features and the response variable
y = df['avg_pupil_size'].values
X = df.drop('avg_pupil_size', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
_ = knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

##############################################################

#%%
# ############## Measuring the Model Performance ##############
# ####### Again here you would have a label type value error : 'continuous'

# rest of this block is based on the sci-kit- inbuilt digit recognition dataset ########

# Import the necessary modules if you hadn't already - see top of script for this

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

#########################################################################
#%%
# "Train/Test Split + Fit/Predict/Accuracy"

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

#########################################################################
#%%
# ############ Over fitting and Underfitting #################

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

##############################################################
#%%
##############################################################
##############################################################
# ################ 2 - Regression ################## #
# ########## Importing Data for Supervised Learning ##########

# Read the CSV file into a DataFrame: df
df2 = pd.read_csv('fixations.csv', index_col=0)

print(df2.head())

#%%
# Create arrays for features and target variable
y = df2['duration'].values
X = df2['avg_pupil_size'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1, 1)
X = X.reshape(-1, 1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

################################################################
#%%
# ###### Explorig the Data using Seaborn Heatmap ####
sns.heatmap(df2.corr(), square=True, cmap='RdYlGn')
# All features shown but not necessary

################################################################
#%%
################################################################
# ############# Fit and Predict for Regression # ###############

# Create the regressor model: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(df2['duration'].values),
                               max(df2['duration'].values)).reshape(-1, 1)

# Fit the model to the data
reg.fit(X, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print('The r_squared regression score is ', reg.score(X, y))

#%%
# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

########################################################################
#%%
# ############## Train/Test Data Split for Regression Analysis #########
# it is imperative that our SLM has the ablity to gen. well for new data
# True for class. models and reg models alike - Reg Ex on all features

# Import necessary modules
# from sklearn.linear_model import LinearRegression

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

###########################################################################
#%%
# ####Basic k-fold Cross Val. (k value Proportional to Resource reqmnt)

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

##########################################################################
#%%
# ###### k-fold cv comparisons #######

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print('The "np.mean" 3 fold CV value is ', np.mean(cvscores_3))

# Perform 5-fold CV
cvscores_5 = cross_val_score(reg, X, y, cv=5)
print('The "np.mean" 5 fold CV value is ', np.mean(cvscores_5))

# Perform 7-fold CV
cvscores_7 = cross_val_score(reg, X, y, cv=7)
print('The "np.mean" 7 fold CV value is ', np.mean(cvscores_7))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print('The "np.mean" 10 fold CV value is ', np.mean(cvscores_10))

#########################################################################
#%%
# we could use the timeit method to check resource reqt

# %timeit cross_val_score(reg, X, y, cv = ____)

##########################################################################
##########################################################################
#%%
# ##Regularized Regression Models - Tempering the Data (Over/Under fitting)
# This part also won't work here - see error code and fix to your use case
#  Lasso Regularization - define df_columns

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)    # name_error here - fixable
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()

##############################################################################
#%%
#    Ridge Regularization - 

# Func to plot R.sqrd score and std.error for range of alphas
# Regression is fit over different alphas and CV r.sqrd are plotted

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error,
    cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)



#########################################################################
#%%
#########################################################################
# ################# 3 - Fine Tuning the Model ###################
# ############### Metrics for Classification ############### #

# here (this cell) you will find:
# ValueError: Unknown label type: 'continuous'

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4,
                                                    random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print('The "knn" confusion matrix: ', confusion_matrix(y_test, y_pred))
print('The "knn" classf. report: ', classification_report(y_test, y_pred))
\
print('These numbers would actually make sense if the data was fit for purpose')

##############################################################################
#%%
# ############ Building a Logistic Regression Model ################ #

# here (this cell) you will find:
# ValueError: Unknown label type: 'continuous'

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4,
                                                    random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

######################################################################
# ############# Plotting the roc_curve  #############

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#########################################################################
# ############# AUC Computation #############

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc - 5 fold cross val in this case
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

############################################################################
############################################################################
#%%
# ########## HyperParameter Tuning with GridSearch ########

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))

#############################################################################
#%%
# #### Randomized SearchCV - HyperParam. Tuning - eco computationality

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

##########################################################################
##########################################################################
##########################################################################
#%%
# ########### Hold out set in Practice - Claasification - Stage 1 ########

# Import necessary modules
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=42)


# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

###########################################################################
#%%
# ########  Regression - Part 2 ####

# from sklearn.linear_model import ElasticNet
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=42)


# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)

print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))

###########################################################################
#%%
###########################################################################
# ################# Chapter 4 - Preprocessing & Pipelines ##################

# We explore some categorical features - EDA

# Read whatever 'somefileyouhave.csv' into a DataFrame: df

df3 = pd.read_csv('fixations.csv')    # again fixations are of type continuous

print(df3.head())

print(df3.info)
print(df3.describe)
# Create a boxplot of whatever...
df.boxplot('duration', 'avg_pupil_size', rot=60)

# Show the plot
plt.show()

#########################################################################
# ###### Create dummy variables to cater for the non-int feature ###
#%%
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)

############################################################################
# ##### Regression with Categorical Features ########

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)

############################################################################
# ###### Droping the missing data - CAUTION - could lose useful data

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Df shape After Dropping All Rows c Missing Vals: {}".format(df.shape))

#####################################################################
# NOTE:

# To avoid the pitfalls of losing potentially good data by dropping
# missing values - it is better option to develop an imputation strategy
# Domain knowledge is crucial here.
# Absent domain specific knowledge a goo dimputation strategy would be to
# use the mean/median of the row or column with the missing data

##########################################################################
#%%
##########################################################################

# Imputing Missing Data in a ML PipeLine - Part 1
# Import the Imputer module
# from sklearn.preprocessing import Imputer
# from sklearn.svm import SVC
# SVC is Support Vector Classification- type of SVM

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
         ('SVM', clf)]

# Once the pipeline is set up - we now use this for some classification task

# Imputing missing data in a ML Pipeline - Part 2

# Import necessary modules
# from sklearn.preprocessing import Imputer
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN',
                                strategy='most_frequent',
                                axis=0)), ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))

#############################################################################
# #####Centre and Scale the Data to account for dimension ranges

# Import scale
# from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X)))
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

##########################################################################
# Centering and Scaling in a Pipeline - 2

# Import the necessary modules
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

#############################################################################
############################################################################
#%%
# #### Building a Classification Pipeline - Part 1

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C': [1, 10, 100],
              'SVM__gamma': [0.1, 0.01]}

# Create train and test sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

##########################################################################
#%%
# Pipeline for Regression  - Bringing all togethr - Part 2

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio': np.linspace(0, 1, 30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)

print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

##########################################################################
##########################################################################