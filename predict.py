import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


from sklearn.neighbors import KNeighborsClassifier

# load train set
train_set = pd.read_csv('prepared_train.csv')
X_train = train_set.drop(columns=['Vote'])
y_train = train_set['Vote']

# build models
knn_models = []
# for i in range(1,10):
#     model = KNeighborsClassifier(n_neighbors = i)
#     knn_models.append(model)
#     accuracy = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)
#     print("Accuracy of KNN with k =", i,  " with cross-validation:", accuracy.mean())
#     clf = tree.DecisionTreeClassifier()

gnb = GaussianNB()
accuracy = cross_val_score(gnb, X_train, y_train, scoring='accuracy', cv=10)
print("Accuracy of naive bayes on train test with cross-validation:", accuracy.mean())
svm = svm.SVC(gamma='scale')
accuracy = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=10)
print("Accuracy of SVM on train test with cross-validation:", accuracy.mean())
clf = tree.DecisionTreeClassifier(criterion='gini')
accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=10)
print("Accuracy of Decision Tree on train test with cross-validation:", accuracy.mean())


# load validation set
validation_set = pd.read_csv('prepared_validation.csv')
X_validation = validation_set.drop(columns=['Vote'])
y_validation = validation_set['Vote']

# KNN k=1 check
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_validation)
print("classification report of knn k=1:")
print(classification_report(y_validation, knn_pred))

# NB check
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_validation)
print("classification report of naive bayes:")
print(classification_report(y_validation, gnb_pred))

# svm check
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_validation)
print("classification report of SVM:")
print(classification_report(y_validation, svm_pred))

# decision tree check
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_validation)
print("classification report of decision tree:")
print(classification_report(y_validation, clf_pred))

best_model = clf
