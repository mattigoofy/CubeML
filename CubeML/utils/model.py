import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import loguniform
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import typing


def load_dataset(filepath: str, use_n: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pandas dataframe dataset from pickle.

    Args:
        filepath (str): filepath to the dataset to use.
        use_n (int|None): if set, only use the first n rows of the dataset. If None, use all rows.

    Returns:
        X, y (tuple[DataFrame, DataFrame]): tuple of data and labels

    _______________DATA__________________    ___LABEL___
    TILE_R1     TILE_R2 ...     TILE_B9         MOVE
    0           1               3               R           --> state 1
    4           2               5               L           --> state 2
    """
    df = pd.read_pickle(filepath)
    moves_df = df["MOVE"]
    data_df = df.loc[:, df.columns != "MOVE"]
    if use_n:
        moves_df = moves_df[0:use_n]
        data_df = data_df[0:use_n]
    return (data_df, moves_df)


def train_model(classifier: typing.Literal["decision_tree", "mlp"], filepath: str = "cfop-dataset-processed/dataset.pkl", use_n: int | None = None) -> tuple[typing.Any, pd.DataFrame, pd.DataFrame]:
    match classifier:
        case "decision_tree":
            return train_model_decision_tree(filepath, use_n)
        case "mlp":
            return train_model_mlp(filepath, use_n)


def train_model_decision_tree(filepath: str = "cfop-dataset-processed/dataset.pkl", use_n: int | None = None) -> tuple[typing.Any, pd.DataFrame, pd.DataFrame]:
    """
    Train a new model using the specified dataset.

    Args:
        filepath (str): filepath to the dataset to use.
        use_n (int|None): if set, only use the first n rows of the dataset. If None, use all rows.

    Returns:
        The new model, along with X_test and y_test.
    """
    # TODO change filepath to actual file
    X, y = load_dataset(filepath, use_n)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"],
    }

    classifier = RandomForestClassifier(random_state=42)

    grid_search = sklearn.model_selection.RandomizedSearchCV(
        classifier,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    return (grid_search, X_test, y_test)


def train_model_mlp(filepath: str = "cfop-dataset-processed/dataset.pkl", use_n: int | None = None) -> tuple[typing.Any, pd.DataFrame, pd.DataFrame]:
    """
    Train a new model using the specified dataset.

    Args:
        filepath (str): filepath to the dataset to use.
        use_n (int|None): if set, only use the first n rows of the dataset. If None, use all rows.

    Returns:
        The new model, along with X_test and y_test.
    """
    X, y = load_dataset(filepath, use_n)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

    param_distributions = {
        # Network architecture
        "hidden_layer_sizes": [
            (256, 128),
            (512, 256),
            (512, 256, 128),
            (256, 256, 128),
            (1024, 512, 256),
            (512, 512, 256, 128),
        ],

        # Activation
        "activation": ["relu", "tanh"],

        # Solver & learning rate
        "solver": ["adam"],
        "learning_rate_init": loguniform(1e-4, 1e-2),
        "learning_rate": ["constant", "adaptive"],

        # Regularisation
        "alpha": loguniform(1e-5, 1e-1),     # L2 penalty

        # Stopping
        "max_iter": [300, 500],
        # "early_stopping": [True],
        "validation_fraction": [0.1],
        # "n_iter_no_change": [20],
    }

    classifier = MLPClassifier(random_state=42)

    grid_search = sklearn.model_selection.RandomizedSearchCV(
        classifier,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    return (grid_search, X_test, y_test)


def show_model_score(grid_search, X_test, y_test):
    """
    Gets the ROC curve for the given model.

    Args:
        grid_search: The trained model.
        X_test: DataFrame of all test data.
        y_test: DataFrame of corresponding labels.
    """
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
    model, X_test, y_test = train_model_decision_tree()
    show_model_score(model, X_test, y_test)

    random_state = X_test.iloc[0]
    prediction = model.predict((random_state,))

    print(prediction)
