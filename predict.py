import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] )  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print(" " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("%{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}s".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


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


parties = y_validation.values
parties = np.unique(parties)

# KNN k=1 check
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_validation)
print("classification report of knn k=1:")
print(classification_report(y_validation, knn_pred))
cm = confusion_matrix(y_validation, knn_pred)
print_cm(cm, parties)

# NB check
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_validation)
print("classification report of naive bayes:")
print(classification_report(y_validation, gnb_pred))
cm = confusion_matrix(y_validation, gnb_pred)
print_cm(cm, parties)

# svm check
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_validation)
print("classification report of SVM:")
print(classification_report(y_validation, svm_pred))
cm = confusion_matrix(y_validation, svm_pred)
print_cm(cm, parties)

# decision tree check
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_validation)
print("classification report of decision tree:")
print(classification_report(y_validation, clf_pred))
cm = confusion_matrix(y_validation, clf_pred)
print_cm(cm, parties)

best_model = svm
