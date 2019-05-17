import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot


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


def check_model(trained_model, model_name):
    pred = trained_model.predict(X_validation)
    pred = pd.DataFrame(data=pred.flatten())
    hist = pred[0].value_counts(normalize=True) * 100
    print("classification report of {}:".format(model_name))
    print(classification_report(y_validation, pred))
    cm = confusion_matrix(y_validation, pred)
    print_cm(cm, parties)
    d = hist.to_frame().join(validation_hist)
    d['diff'] = abs(d['Vote'] - d[0])
    print("Histogram compare - {} vs validation:".format(model_name))
    print(d)
    print("total diff =", d['diff'].sum())
    print("correlation:", d['Vote'].corr(d[0]))

# load train
train_set = pd.read_csv('prepared_train.csv')
X_train = train_set.drop(columns=['Vote'])
y_train = train_set['Vote']

# load validation set
validation_set = pd.read_csv('prepared_validation.csv')
X_validation = validation_set.drop(columns=['Vote'])
y_validation = validation_set['Vote']

# get histogram of validation set
parties = np.unique(y_validation.values)
validation_hist = validation_set['Vote'].value_counts(normalize=True) * 100

######
# KNN#
######
# Train the model using CV to find the best option foreach number of neighbors

knn_models = []
for i in range(1,15):
    best_accuracy,best_model = 0, None
    for weight in ['uniform', 'distance']:
        for algo in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            for leaf_size in range(10,100,10):
                model = KNeighborsClassifier(n_neighbors = i,weights=weight, algorithm=algo, leaf_size=leaf_size)
                accuracy = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10).mean()
                if best_accuracy < accuracy:
                    best_accuracy, best_model = accuracy, model
    knn_models.append(best_model)
    print("Best Accuracy of KNN with k ={} with cross-validation:{} - params weights={}, algorithm={}, leaf_size={}".format(i, best_accuracy, best_model.weights, best_model.algorithm,best_model.leaf_size ))

# Validate models and pick up the best one
best_accuracy,best_model = 0, None
for model in knn_models:
    model.fit(X_train, y_train)
    accuracy = model.score(X_validation, y_validation)
    if best_accuracy < accuracy:
        best_accuracy, best_model = accuracy, model
    print("Accuracy of KNN with k ={} on validation set:{}".format(model.n_neighbors, accuracy))

print("Best Accuracy of KNN with k ={} on validation set:{} - params weights={}, algorithm={}, leaf_size={}".format(best_model.n_neighbors, best_accuracy, best_model.weights, best_model.algorithm,best_model.leaf_size ))

check_model(best_model, 'KNN')


#############
# Naive Base#
#############
gnb = GaussianNB()
accuracy = cross_val_score(gnb, X_train, y_train, scoring='accuracy', cv=10)
print("Accuracy of naive bayes on train test with cross-validation:", accuracy.mean())
gnb.fit(X_train, y_train)
nb_score = gnb.score(X_validation, y_validation)
print("Best Accuracy of NB on validation set:{}".format(nb_score))

# NB check
check_model(gnb, 'NaiveBase')


################
# Decision Tree#
################
dt_models = []
for i in range(1,X_train.columns.size + 1):
    best_accuracy,best_model = 0, None
    for criterion in ['gini', 'entropy']:
        for splitter in ['best', 'random']:
                model = DecisionTreeClassifier(max_depth=i, criterion =criterion , splitter =splitter)
                accuracy = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10).mean()
                if best_accuracy < accuracy:
                    best_accuracy, best_model = accuracy, model
    dt_models.append(best_model)
    print("Best Accuracy of Decision Tree with depth ={} with cross-validation:{} - params criterion={}, splitter={}".format(i, best_accuracy, best_model.criterion, best_model.splitter))

# Validate models and pick up the best one
best_accuracy,best_model = 0, None
for model in dt_models:
    model.fit(X_train, y_train)
    accuracy = model.score(X_validation, y_validation)
    if best_accuracy < accuracy:
        best_accuracy, best_model = accuracy, model
    print("Accuracy of Decision Tree with depth ={} on validation set:{}".format(model.max_depth, accuracy))

print("Best Accuracy of Decision Tree with k ={} on validation set:{} - params criterion={}, splitter={}".format(best_model.max_depth, best_accuracy, best_model.criterion, best_model.splitter))

check_model(best_model, 'Decision Tree')

######
# SVM#
######
best_accuracy,best_model = 0, None
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    for gamma in [1 / (X_train.columns.size * X_train.var().mean()), 1 / X_train.columns.size]:
            for shrinking in [True, False]:
                for probability in [True, False]:
                    for decision_function_shape in ['ovo','ovr']:
                        model = svm.SVC(kernel=kernel, gamma=gamma, shrinking=shrinking, probability=probability, decision_function_shape=decision_function_shape)
                        accuracy = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10).mean()
                        if best_accuracy < accuracy:
                            best_accuracy, best_model = accuracy, model
    print('finish kernel ', kernel)
print("Best Accuracy of SVM with cross-validation:{} - params  kernel={},gamma={}, shrinking={}, probability={}, decision_function_shape={}".format(best_accuracy, best_model.kernel, best_model.gamma, best_model.shrinking, best_model.probability, best_model.decision_function_shape))
best_model.fit(X_train, y_train)
svm_score = best_model.score(X_validation, y_validation)
print("Best Accuracy of SVM on validation set - {}".format(svm_score))

# svm check
check_model(best_model, 'SVM')

# Load origin file
df = pd.read_csv('ElectionsData.csv')
print()
parties_by_percents = df['Vote'].value_counts(normalize=True) * 100
print("The division of voters between the various parties:")

