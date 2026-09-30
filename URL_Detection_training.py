#--------------------------------------------------
# PHISHING URL BEHAVIOUR CLASSIFICATION
#--------------------------------------------------

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


#------------------------------------------
#        Importing the Dataset
#------------------------------------------
df = pd.read_csv("Phishing_URL_Dataset.csv")

print(df.head())


#------------------------------------------
#        Basic Dataset Information
#------------------------------------------

print("*"*60)
print("Dataset Shape")
print("*"*60)

print(df.shape)

print("*"*60)
print("Dataset Columns")
print("*"*60)

print(df.columns)

print("*"*60)
print("Dataset Information")
print("*"*60)

print(df.info())

print("*"*60)
print("Data Types of Columns")
print("*"*60)

print(df.dtypes)

print("*"*60)
print("Missing values")
print("*"*60)

print(df.isnull().sum())

print("*"*60)
print("Duplicate records")
print("*"*60)

print(df.duplicated().sum())



#------------------------------------------
#        Target Variable Distribution
#------------------------------------------

print("*"*60)
print("Target Variable Distribution")
print("*"*60)

print(df["label"].value_counts())


#------------------------------------------
#        Statistical Summary
#------------------------------------------

print("*"*60)
print("Statistical Summary")
print("*"*60)

print(df.describe())

#------------------------------------------
#        Data Leakage Check
#------------------------------------------

print("*"*60)
print("Data Leakage Check")
print("*"*60)

print("Target Variable:", "label")
print("Features containing 'label' or target-related names:")

for column in df.columns:
    if "label" in column.lower():
        print(column)


#------------------------------------------
#        Feature Relevance
#------------------------------------------

print("*"*60)
print("Feature Relevance")
print("*"*60)

correlation = df.corr(numeric_only=True)["label"].sort_values(ascending=False)

print(correlation)


#----------------------------------------------
#     Separating Features(X) and Target(y)
#----------------------------------------------

X = df.drop("label", axis=1)
Y = df["label"]

print("*"*60)
print("Features Shape")
print("*"*60)

print(X.shape)

print("*"*60)
print("Target Shape")
print("*"*60)

print(Y.shape)


#------------------------------------------
#        Selecting URL Based Features
#------------------------------------------

url_features = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS"
]

X = X[url_features]

print("*"*60)
print("URL Based Features")
print("*"*60)

print(X.columns)
print("Number of Features:", len(X.columns))


#------------------------------------------
#        Splitting Dataset
#------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("*"*60)
print("Training and Testing Data")
print("*"*60)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test:", Y_test.shape)

#------------------------------------------
#        Feature Scaling
#------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#------------------------------------------
#        Logistic Regression
#------------------------------------------

model_lr = LogisticRegression(max_iter=1000)

model_lr.fit(X_train, Y_train)

Y_pred_lr = model_lr.predict(X_test)

        
#------------------------------------------
#        Logistic Regression Evaluation
#------------------------------------------

accuracy_lr = accuracy_score(Y_test, Y_pred_lr)

print("*"*60)
print("Logistic Regression Accuracy")
print("*"*60)

print("Accuracy:", accuracy_lr)


#------------------------------------------
#        Confusion Matrix
#------------------------------------------

cm_lr = confusion_matrix(Y_test, Y_pred_lr)

print("*"*60)
print("Confusion Matrix")
print("*"*60)

print(cm_lr)


precision_lr = precision_score(Y_test, Y_pred_lr)
recall_lr = recall_score(Y_test, Y_pred_lr)
f1_lr = f1_score(Y_test, Y_pred_lr)

print("*"*60)
print("Logistic Regression Metrics")
print("*"*60)

print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1 Score:", f1_lr)


#------------------------------------------
#        ROC-AUC Score
#------------------------------------------

Y_prob_lr = model_lr.predict_proba(X_test)[:, 1]

roc_auc_lr = roc_auc_score(Y_test, Y_prob_lr)

print("*"*60)
print("ROC-AUC Score")
print("*"*60)

print("ROC-AUC:", roc_auc_lr)


#------------------------------------------
#        K-Nearest Neighbors
#------------------------------------------

model_knn = KNeighborsClassifier()

model_knn.fit(X_train, Y_train)

Y_pred_knn = model_knn.predict(X_test)

#------------------------------------------
#        KNN Evaluation
#------------------------------------------

accuracy_knn = accuracy_score(Y_test, Y_pred_knn)

print("*"*60)
print("KNN Accuracy")
print("*"*60)

print("Accuracy:", accuracy_knn)


#------------------------------------------
#        KNN Confusion Matrix
#------------------------------------------

cm_knn = confusion_matrix(Y_test, Y_pred_knn)

print("*"*60)
print("KNN Confusion Matrix")
print("*"*60)

print(cm_knn)


#------------------------------------------
#        KNN Metrics
#------------------------------------------

precision_knn = precision_score(Y_test, Y_pred_knn)
recall_knn = recall_score(Y_test, Y_pred_knn)
f1_knn = f1_score(Y_test, Y_pred_knn)

print("*"*60)
print("KNN Metrics")
print("*"*60)

print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1 Score:", f1_knn)


#------------------------------------------
#        KNN ROC-AUC Score
#------------------------------------------

Y_prob_knn = model_knn.predict_proba(X_test)[:, 1]

roc_auc_knn = roc_auc_score(Y_test, Y_prob_knn)

print("*"*60)
print("KNN ROC-AUC Score")
print("*"*60)

print("ROC-AUC:", roc_auc_knn)

#------------------------------------------
#        Model Comparison
#------------------------------------------

print("*"*60)
print("Model Comparison")
print("*"*60)

print("Logistic Regression")
print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1 Score:", f1_lr)
print("ROC-AUC:", roc_auc_lr)

print("*"*60)

print("KNN")
print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1 Score:", f1_knn)
print("ROC-AUC:", roc_auc_knn)

#------------------------------------------
#        Error Analysis
#------------------------------------------

print("*"*60)
print("Error Analysis")
print("*"*60)

print("Logistic Regression Errors:",
      (Y_test != Y_pred_lr).sum())

print("KNN Errors:",
      (Y_test != Y_pred_knn).sum())

#------------------------------------------
#        Error Type Analysis
#------------------------------------------

print("*"*60)
print("Error Type Analysis")
print("*"*60)

print("Logistic Regression")
print("False Positives:", cm_lr[0][1])
print("False Negatives:", cm_lr[1][0])

print("*"*60)

print("KNN")
print("False Positives:", cm_knn[0][1])
print("False Negatives:", cm_knn[1][0])


#------------------------------------------
#        Phishing Class Metrics
#------------------------------------------

print("*"*60)
print("Phishing Class Metrics")
print("*"*60)

print("Logistic Regression")
print("Precision:", precision_score(Y_test, Y_pred_lr, pos_label=0))
print("Recall:", recall_score(Y_test, Y_pred_lr, pos_label=0))
print("F1 Score:", f1_score(Y_test, Y_pred_lr, pos_label=0))

print("*"*60)

print("KNN")
print("Precision:", precision_score(Y_test, Y_pred_knn, pos_label=0))
print("Recall:", recall_score(Y_test, Y_pred_knn, pos_label=0))
print("F1 Score:", f1_score(Y_test, Y_pred_knn, pos_label=0))

#------------------------------------------
#        Saving URL Model
#------------------------------------------

feature_names = X.columns.tolist()

print("*"*60)
print("Features Used by the URL Model")
print("*"*60)

print(feature_names)

joblib.dump(feature_names, "phishing_url_features.pkl")

#------------------------------------------
#        Saving Model and Scaler
#------------------------------------------

joblib.dump(model_lr, "phishing_url_model.pkl")
joblib.dump(scaler, "phishing_url_scaler.pkl")



































































































