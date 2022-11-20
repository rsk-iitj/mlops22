import argparse
import numpy as np
from sklearn import datasets, metrics, svm
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from utils import data_preprocess
from utils import train_dev_test_split
from utils import h_param_tuning

parser = argparse.ArgumentParser()
parser.add_argument('--clf_name', type=str)
parser.add_argument('--random_state', type=int)
args = parser.parse_args()



digits = datasets.load_digits()

data, label = data_preprocess(digits)

del digits

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.6, 1, 2, 5, 7, 10]
h_param_comb_svm = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]

min_samples_split_list = [2,3,5,10]
min_samples_leaf_list = [1,3,5,10]
h_param_comb_dtree = [{"min_samples_leaf": g, "min_samples_split": c} for g in min_samples_leaf_list for c in min_samples_split_list]

model_of_choices = [svm.SVC(),tree.DecisionTreeClassifier()]
hp_of_choices = [h_param_comb_svm,h_param_comb_dtree]
metric=metrics.accuracy_score

x = list(set(label))

split_frac = [.6,]
dev_frac = [.1,]


if args.clf_name in "svm":
    clf = model_of_choices[0]
elif args.clf_name in "tree":
    clf = model_of_choices[1]
X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
    data, label, split_frac[0],dev_frac[0],args.random_state
)
best_model, best_metric, best_h_params = h_param_tuning(hp_of_choices[0], clf, X_train, y_train, X_dev, y_dev, X_test, y_test, metric)
prediction = best_model.predict(X_test)
cm = metrics.confusion_matrix(y_test, prediction)
print("\n")
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, prediction)}\n"
)
print(
    f"Confusion report for classifier {clf}:\n"
    f"{metrics.confusion_matrix(y_test, prediction)}\n"
)
