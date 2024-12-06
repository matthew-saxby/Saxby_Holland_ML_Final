import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

diabetes_processed = pd.read_csv('diabetes_processed.csv')

X = diabetes_processed[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]# Features
y = diabetes_processed['diabetes']# labels

#set random seed
np.random.seed(123)

# Split the data into 80% train/validation and 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# Split the 80% train/validation into 70% train and 10% validation
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=123)


#uncomment to see the shape of the data
#print('Training:', X_train.shape)
#print('Testing:', X_test.shape)
#print('Training:', X_val.shape)


# Train the model

#set features
features = ['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']
#set depth limit
depth_limit = 5

#create a model
model = DecisionTreeClassifier(criterion= 'entropy',max_depth=depth_limit)
#fit the model
model.fit(X_train[features],y_train)

#predict on train data 
y_pred_train = model.predict(X_train[features])

#predict on the test data
y_pred_test = model.predict(X_test[features])

#Evaluate the train model
train_accuracy = metrics.accuracy_score(y_train,y_pred_train)
train_f1 = metrics.f1_score(y_train, y_pred_train)

#Evaluate the test model
test_accuracy = metrics.accuracy_score(y_test,y_pred_test)
test_f1 = metrics.f1_score(y_test,y_pred_test)

print('----Training performance----')
print('Train Accuracy:', train_accuracy)
print('Train F1 Score', train_f1)
print('----Testing performance----')
print('Test Accuracy:', test_accuracy)
print('Test F1 Score', test_f1)


#plot the decision tree

plt.figure(figsize=(12,8))
plot_tree(model,feature_names=features,class_names=['0','1'],filled=True)
plt.title(f'Decision Tree (Features: {features}, Max Depth: {depth_limit})')
plt.show()