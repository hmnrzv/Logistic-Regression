# Loading and exploring dataset
import pandas as pd
#Reading the file into a dataframe
PATH = '/content/drive/MyDrive/5th semester/AI/LABS/LAB5'
data=pd.read_csv(f'{PATH}/diabetes_Modified.csv')

#Displaying the read contents
data

# separating predictors and target
X = data.drop("Outcome",axis=1) #predictors
Y = data["Outcome"]  #target

# Splitting into train and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y,test_size=0.20,random_state=0)

# Create logistic regression object
%%time
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
logistic_regression1 = LogisticRegression(solver="liblinear", random_state=10)
# Train model
model1 = logistic_regression1.fit(X_train, Y_train)
# This gives probabilities for both class 0 and class 1
probabilities = model1.predict_proba(X_train)
# first col shows prob for class 0 and second col for class 1

# Use this if We only need the probabilities for the positive class (class 1)
positive_probabilities = probabilities[:, 1]
print (probabilities)
print (positive_probabilities)

#MODEL 1
custom_threshold = 0.5
Y_pred_train_2 = (positive_probabilities >= custom_threshold).astype(int)
# Printing results
print(confusion_matrix(Y_train, Y_pred_train_2))
print("Accuracy: ",metrics.accuracy_score(Y_train,Y_pred_train_2))
print('Precision: ',metrics.precision_score(Y_train,Y_pred_train_2))
print('Recall score: ',metrics.recall_score(Y_train,Y_pred_train_2))
print('F1 score: ',metrics.f1_score(Y_train,Y_pred_train_2))

#MODEL 2
custom_threshold = 0.4
Y_pred_train_2 = (positive_probabilities >= custom_threshold).astype(int)
# Printing results
print(confusion_matrix(Y_train, Y_pred_train_2))
print("Accuracy: ",metrics.accuracy_score(Y_train,Y_pred_train_2))
print('Precision: ',metrics.precision_score(Y_train,Y_pred_train_2))
print('Recall score: ',metrics.recall_score(Y_train,Y_pred_train_2))
print('F1 score: ',metrics.f1_score(Y_train,Y_pred_train_2))

#MODEL 3
custom_threshold = 0.75
Y_pred_train_2 = (positive_probabilities >= custom_threshold).astype(int)
# Printing results
print(confusion_matrix(Y_train, Y_pred_train_2))
print("Accuracy: ",metrics.accuracy_score(Y_train,Y_pred_train_2))
print('Precision: ',metrics.precision_score(Y_train,Y_pred_train_2))
print('Recall score: ',metrics.recall_score(Y_train,Y_pred_train_2))
print('F1 score: ',metrics.f1_score(Y_train,Y_pred_train_2))


#Performance Evaluaton on Test Set
#Predictions
Y_pred_test = model1.predict(X_test)

# Printing results
print(confusion_matrix(Y_test, Y_pred_test))
print("Accuracy: ",metrics.accuracy_score(Y_test,Y_pred_test))
print('Precision: ',metrics.precision_score(Y_test,Y_pred_test))
print('Recall score: ',metrics.recall_score(Y_test,Y_pred_test))
print('F1 score: ',metrics.f1_score(Y_test,Y_pred_test))