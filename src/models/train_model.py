import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from evaluate_model import printPerformance, printTestPerformance
from sklearn.neighbors import KNeighborsClassifier
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score


diabetes_processed = pd.read_csv('diabetes_processed.csv')

X = diabetes_processed[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]# Features
y = diabetes_processed['diabetes']# labels

#set random seed
np.random.seed(123)

# Split the data into 70% train and 30% validation/test
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=123)
# Split the 70% train/validation into 66% train and 33% validation
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.66666, random_state=123)


#uncomment to see the shape of the data
#print('Training:', X_train.shape)
#print('Validation:', X_val.shape)
#print('Testing:', X_test.shape)

# Train the model

#set features
features = ['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']



### Decision Tree Classifier ###

#set depth limit
depth_limit = 8

#create a model
model = DecisionTreeClassifier(criterion= 'entropy',max_depth=depth_limit)
#fit the model
model.fit(X_train[features],y_train)

#predict on train data 
dt_y_pred_train = model.predict(X_train[features])

#predict on the validation data
dt_y_pred_val = model.predict(X_val[features])
dt_y_pred_test = model.predict(X_test[features])


#Evaluate the train model
dt_train_accuracy = metrics.accuracy_score(y_train,dt_y_pred_train)
dt_train_f1 = metrics.f1_score(y_train, dt_y_pred_train)

#Evaluate the test model
dt_val_accuracy = metrics.accuracy_score(y_val,dt_y_pred_val)
dt_val_f1 = metrics.f1_score(y_val,dt_y_pred_val)

#Evaluate the test model
dt_test_accuracy = metrics.accuracy_score(y_test,dt_y_pred_test)
dt_test_f1 = metrics.f1_score(y_test,dt_y_pred_test)


print("----DT Performance----")
print("")
print("")
printPerformance(dt_train_accuracy, dt_train_f1, dt_val_accuracy, dt_val_f1)

printTestPerformance(dt_val_accuracy, dt_val_f1, dt_test_accuracy, dt_test_f1)



# Predict probabilities for the test set
val_prob = model.predict_proba(X_val[features])[:, 1]  # Probabilities for the positive class

fpr, tpr, thresholds = metrics.roc_curve(y_val, val_prob)
roc_auc = metrics.roc_auc_score(y_val, val_prob)

'''
# Compute ROC-AUC and plot ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_val_prob)
roc_auc = metrics.roc_auc_score(y_val, y_val_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()
'''



### KNN ###

knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)

# Predict on train and validation data
knn_y_pred_train = knn_model.predict(X_train)
knn_y_pred_val = knn_model.predict(X_val)

# Evaluate the train model
knn_train_accuracy = metrics.accuracy_score(y_train, knn_y_pred_train)
knn_train_f1 = metrics.f1_score(y_train, knn_y_pred_train)

# Evaluate the validation model
knn_val_accuracy = metrics.accuracy_score(y_val, knn_y_pred_val)
knn_val_f1 = metrics.f1_score(y_val, knn_y_pred_val)

# Print KNN performance
print("")
print("")
print("----KNN Performance----")
printPerformance(knn_train_accuracy, knn_train_f1, knn_val_accuracy, knn_val_f1)


# Predict probabilities for the validation set
knn_val_prob = knn_model.predict_proba(X_val)[:, 1]  # Probabilities for the positive class

fpr_knn, tpr_knn, thresholds_nb = metrics.roc_curve(y_val, knn_val_prob)
roc_auc_knn = metrics.roc_auc_score(y_val, knn_val_prob)

'''
# Compute ROC-AUC and plot ROC curve for KNN
fpr_knn, tpr_knn, thresholds_knn = metrics.roc_curve(y_val, knn_val_prob)
roc_auc_knn = metrics.roc_auc_score(y_val, knn_val_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label=f'KNN ROC curve (AUC = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - KNN')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()
'''

### Naive Bayes Classifier ###

# Initialize and train the model
NB = GaussianNB()
NB.fit(X_train, y_train)

# Make predictions
nb_y_pred_train = NB.predict(X_train)
nb_y_pred_val = NB.predict(X_val)

# Evaluate the train model
nb_train_accuracy = metrics.accuracy_score(y_train, nb_y_pred_train)
nb_train_f1 = metrics.f1_score(y_train, nb_y_pred_train)

# Evaluate the validation model
nb_val_accuracy = metrics.accuracy_score(y_val, nb_y_pred_val)
nb_val_f1 = metrics.f1_score(y_val, nb_y_pred_val)

print("")
print("")
print("----NB Performance----")
printPerformance(nb_train_accuracy, nb_train_f1, nb_val_accuracy, nb_val_f1)

# Predict probabilities for the validation set
nb_val_prob = NB.predict_proba(X_val)[:, 1]  # Probabilities for the positive class

# Compute ROC-AUC and plot ROC curve for Naive Bayes
fpr_nb, tpr_nb, thresholds_nb = metrics.roc_curve(y_val, nb_val_prob)
roc_auc_nb = metrics.roc_auc_score(y_val, nb_val_prob)
'''
# Plotting the ROC curve for Naive Bayes
plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, color='red', lw=2, label=f'Naive Bayes ROC curve (AUC = {roc_auc_nb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Naive Bayes')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()
'''

# Create a new figure for the combined ROC curve
plt.figure(figsize=(10, 8))

# Plot Decision Tree ROC Curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Decision Tree ROC curve (AUC = {roc_auc:.2f})')

# Plot KNN ROC Curve
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label=f'KNN ROC curve (AUC = {roc_auc_knn:.2f})')

# Plot Naive Bayes ROC Curve
plt.plot(fpr_nb, tpr_nb, color='red', lw=2, label=f'Naive Bayes ROC curve (AUC = {roc_auc_nb:.2f})')

# Plot the diagonal line (chance level)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set limits and labels
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()
