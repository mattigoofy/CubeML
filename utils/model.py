import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import typing


def load_dataset(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pandas dataframe dataset from pickle

    Outputs:
        X: data
        y: labels

    _______________DATA__________________    ___LABEL___
    TILE_R1     TILE_R2 ...     TILE_B9         MOVE
    0           1               3               R           --> state 1
    4           2               5               L           --> state 2
    """
    df = pd.read_pickle(filepath)
    moves_df = df["MOVE"]
    data_df = df.loc[:, df.columns != "MOVE"]
    return (data_df, moves_df)


def train_model(filepath: str = "cfop-dataset-processed/dataset.pkl") -> tuple[typing.Any, pd.DataFrame, pd.DataFrame]:
    """
    Train a new model using the specified dataset.

    Returns:
        The new model, along with X_test and y_test.
    """
    # TODO change filepath to actual file
    X, y = load_dataset(filepath)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

    param_grid = {
        'max_depth': np.arange(2, 5, 1)
    }

    # param_grid = {
    #     'n_estimators': ...,
    #     'max_depth': ...,
    #     'max_features': ...,
    #     'min_samples_split': ...,
    #     'min_samples_leaf': ...,
    # }

    classifier = RandomForestClassifier()

    # TODO: different scoring algoritme?
    grid_search = sklearn.model_selection.RandomizedSearchCV(classifier, param_grid, cv=2, n_iter=50, scoring='f1_macro') # Of: scoring='roc_auc'
    grid_search.fit(X_train, y_train)

    return (grid_search, X_test, y_test)


def show_model_score(grid_search, X_test, y_test):
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    # Plot the ROC curve (one per class)
    y_score = grid_search.predict_proba(X_test)

    # Binarize y_test for OvR comparison
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)  # shape: (n_samples, n_classes)
    classes = lb.classes_

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    for i, cls in enumerate(classes):
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test_bin[:, i], y_score[:, i])
        auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.show()

    # Multiclass AUC-ROC score
    auc_score = sklearn.metrics.roc_auc_score(y_test, y_score, multi_class='ovr')
    print(f"AUC score: {auc_score}")


if __name__ == '__main__':
    model, X_test, y_test = train_model()
    show_model_score(model, X_test, y_test)

    random_state = X_test.iloc[0]
    prediction = model.predict((random_state,))

    print(prediction)
