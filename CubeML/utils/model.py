import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import loguniform
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


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


def train_model(classifier: typing.Literal["random_forest", "mlp", "encoder", "autoencoder", "mlp_fixed"], filepath: str = "cfop-dataset-processed/dataset.pkl", use_n: int | None = None) -> tuple[typing.Any, pd.DataFrame, pd.DataFrame]:
    match classifier:
        case "random_forest":
            return train_model_random_forest(filepath, use_n)
        case "mlp":
            return train_model_mlp(filepath, use_n)
        case "encoder":
            return train_encoder(filepath, use_n)
        case "autoencoder":
            return train_autoencoder_with_classifier(filepath, use_n)
        case "mlp_fixed":
            return train_model_mlp_fixed(filepath, use_n)


def train_model_random_forest(filepath: str = "cfop-dataset-processed/dataset.pkl", use_n: int | None = None) -> tuple[typing.Any, pd.DataFrame, pd.DataFrame]:
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
        "n_estimators": [50, 80, 100, 200, 300, 400, 500],
        "max_depth": [10, 20, 30, None],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced", "balanced_subsample"],
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
            (256,) * 4,  # uniform depth
        ],

        # Activation
        "activation": ["relu", "tanh"],

        # Solver & learning rate
        "solver": ["adam"],
        "learning_rate_init": loguniform(1e-4, 1e-2),
        "learning_rate": ["constant", "adaptive"],

        # Regularisation
        "alpha": loguniform(1e-5, 1e-1),  # L2 penalty
        # Stopping
        "max_iter": [300, 500],
        # "max_iter": [1000, 2000],
        # "early_stopping": [True],
        "validation_fraction": [0.1],
        # "n_iter_no_change": [20, 30],
        "batch_size": [64, 128, 256],
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


def train_encoder(
    filepath: str = "cfop-dataset-processed/dataset.pkl", use_n: int | None = None
) -> tuple[typing.Any, pd.DataFrame, pd.DataFrame]:
    """
    Train an encoder using TensorFlow/Keras to learn compressed representations of cube states.
    
    Args:
        filepath (str): filepath to the dataset to use.
        use_n (int|None): if set, only use the first n rows of the dataset. If None, use all rows.
    
    Returns:
        The trained encoder model, along with X_test and y_test.
    """
    X, y = load_dataset(filepath, use_n)
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )
    
    class KerasEncoderClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, encoder_layers=(256, 128, 64), epochs=50, batch_size=128):
            self.encoder_layers = encoder_layers
            self.epochs = epochs
            self.batch_size = batch_size
            self.model = None
        
        def fit(self, X, y):
            self._le = LabelEncoder()
            y_int = self._le.fit_transform(y)
            self.classes_ = self._le.classes_
            
            self.model = keras.Sequential([
                layers.Input(shape=(X.shape[1],)),
                *[layers.Dense(u, activation="relu") for u in self.encoder_layers],
                layers.Dense(len(self.classes_), activation="softmax")
            ])
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            self.model.fit(X, y_int, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            return self
        
        def predict(self, X):
            probs = self.model.predict(X, verbose=0)
            indices = np.argmax(probs, axis=1)
            return self._le.inverse_transform(indices)   # use the fitted local one

        def predict_proba(self, X):
            return self.model.predict(X, verbose=0)

        def score(self, X, y):
            return np.mean(self.predict(X) == y)
    
    param_grid = {
        "encoder_layers": [(256, 128, 64), (512, 256, 128), (128, 64, 32)],
        "epochs": [30, 50],
        "batch_size": [64, 128],
    }
    
    grid_search = sklearn.model_selection.GridSearchCV(
        KerasEncoderClassifier(),
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro", # Or "f1_macro"
        n_jobs=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return (grid_search, X_test, y_test)


    return (grid_search, X_test, y_test)


def train_autoencoder_with_classifier(
    filepath: str = "cfop-dataset-processed/dataset.pkl", use_n: int | None = None
) -> tuple[typing.Any, pd.DataFrame, pd.DataFrame]:
    """
    Train an encoder using TensorFlow/Keras to learn compressed representations of cube states.
    Then use a randomforest to train a classifier on this compressed state.
    
    Args:
        filepath (str): filepath to the dataset to use.
        use_n (int|None): if set, only use the first n rows of the dataset. If None, use all rows.
    
    Returns:
        The trained encoder model, along with X_test and y_test.
    """
    X, y = load_dataset(filepath, use_n)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    # Step 1: Build and train the autoencoder
    encoder_layers = (256, 128, 64)
    latent_dim = 32

    inputs = keras.Input(shape=(X_train.shape[1],))
    encoded = inputs
    for units in encoder_layers:
        encoded = layers.Dense(units, activation="relu")(encoded)
    latent = layers.Dense(latent_dim, activation="relu")(encoded)

    decoded = latent
    for units in reversed(encoder_layers):
        decoded = layers.Dense(units, activation="relu")(decoded)
    outputs = layers.Dense(X_train.shape[1], activation="linear")(decoded)

    autoencoder = keras.Model(inputs, outputs)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, verbose=0)

    # Step 2: Wrap encoder as a sklearn transformer
    class KerasEncoderTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, encoder):
            self.encoder = encoder

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return self.encoder.predict(X, verbose=0)

    encoder = keras.Model(inputs, latent)

    # Step 3: GridSearch over pipeline
    pipeline = Pipeline([
        ("encoder", KerasEncoderTransformer(encoder)),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    param_distributions = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [10, 20, 30, None],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
    }

    grid_search = sklearn.model_selection.GridSearchCV(
        pipeline,
        param_grid=param_distributions,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    return (grid_search, X_test, y_test)


def train_model_mlp_fixed(filepath: str = "cfop-dataset-processed/dataset.pkl", use_n: int | None = None) -> tuple[typing.Any, pd.DataFrame, pd.DataFrame]:
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

    classifier = MLPClassifier(
        activation="relu",
        alpha=0.0275,
        hidden_layer_sizes=(1024, 512, 256),
        learning_rate="constant",
        learning_rate_init=0.0001349,
        max_iter=1500,
        solver="adam",
        validation_fraction=0.1,
        random_state=42,
    )

    grid_search = sklearn.model_selection.RandomizedSearchCV(
        classifier,
        param_distributions={},
        n_iter=1,
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
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

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
    model, X_test, y_test = train_model_random_forest()
    show_model_score(model, X_test, y_test)

    random_state = X_test.iloc[0]
    prediction = model.predict((random_state,))

    print(prediction)
