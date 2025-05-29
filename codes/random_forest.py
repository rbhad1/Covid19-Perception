from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize

def random_forest_pipeline(X_orig, y, random_seed, param_grid):
    X_train, X_test, y_train, y_test = train_test_split(X_orig, y, test_size = 0.2, train_size = 0.8, random_state=random_seed)

    model = RandomForestClassifier(random_state=random_seed)

    # grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    print("Best Paramaters:", grid_search.best_params_)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    not_random = 0
    pred_prob = model.predict_proba(X_test)
    for item in pred_prob:
        if any(num > 0.5 for num in item):
            not_random+=1
    print(f"Percentage of predictions >50%: {not_random/len(X_test)}")

    # generate confusion matrix
    _  = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize="true")

    # generate classification report
    report = classification_report(y_test, y_pred)
    print(report)

    # generate ROC AUC graph
    print("Model trained on classes:", model.classes_)
    # print("Current encoder classes:", encoder.classes_)

    # get labels and colors
    labels = list(model.classes_)
    colors = ['#A8D5BA', '#FFE299', '#F28C8C']

    # one-hot binarize y_test for ROC
    classes = np.unique(y_test)
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)

    # predict probabilities
    y_proba = model.predict_proba(X_test)

    # calculate ROC and AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f"{labels[i]} AUC: {roc_auc[i]:.2f}")

    # micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f"Micro-average AUC: {roc_auc['micro']:.2f}")
    roc_auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    print("ROC AUC (macro):", roc_auc_macro)

    # plot ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], label=f'{labels[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')

    plt.xlabel("False Positive Rate", fontweight='bold', labelpad=10)
    plt.ylabel("True Positive Rate", fontweight='bold', labelpad=10)
    plt.title("Multi-class ROC Curve (Raw Data)", pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()

    return y_test, y_pred, X_test, pred_prob




