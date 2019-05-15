import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB



from sklearn.neighbors import KNeighborsClassifier

# load validation set
validation_set = pd.read_csv('prepared_validation.csv')
X = validation_set.drop(columns=['Vote'])
y = validation_set['Vote']

# build models
knn_models = []
for i in range(1,10):
    model = KNeighborsClassifier(n_neighbors = i)
    knn_models.append(model)
    accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=10)
    print("Accuracy of KNN with k = ", i,  ": ", accuracy.mean())
    clf = tree.DecisionTreeClassifier()
gnb = GaussianNB()
accuracy = cross_val_score(gnb, X, y, scoring='accuracy', cv=10)
print("Accuracy of naive bayes: ", accuracy.mean())
svm = svm.SVC()
accuracy = cross_val_score(svm, X, y, scoring='accuracy', cv=10)
print("Accuracy of SVM: ", accuracy.mean())
clf = tree.DecisionTreeClassifier()
accuracy = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
print("Accuracy of Decision Tree: ", accuracy.mean())