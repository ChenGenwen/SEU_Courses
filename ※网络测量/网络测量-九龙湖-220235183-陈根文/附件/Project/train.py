 import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import tree 
from sklearn import naive_bayes


def train_func(train_path):
    data = pd.read_csv(train_path)
    X = data.iloc[:, 1:].values
    Y = data.iloc[:, 0].values
    shuffled_X, shuffled_Y = shuffle(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(shuffled_X, shuffled_Y, test_size=0.1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = RandomForestClassifier(n_jobs=-1, random_state=50)
    # classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None)
    # classifier = SVC(C=1.0, kernel='rbf', gamma='auto')
    # classifier = naive_bayes.GaussianNB() 
    score = cross_val_score(classifier, X_train, Y_train)
    print('交叉验证准确度：', str(score.mean()))
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    matrix = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix:")
    print(matrix)
    result = classification_report(Y_test, Y_pred)
    print("Classification Report:")
    print(result)
    Accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", Accuracy)
    joblib.dump(classifier, 'model.pkl')



if __name__ == '__main__':
    train_path = './data/data.csv'
    train_func(train_path)
