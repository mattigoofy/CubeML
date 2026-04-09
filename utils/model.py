import sklearn
import sklearn.ensemble.RandomForestClassifier
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


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
    raise NotImplementedError


# TODO change filepath to actual file
X, y = load_dataset("cfop-dataset-preprocessed/dataset.pickle")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.10, random_state=42)

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

grid_search = sklearn.model_selection.RandomizedSearchCV(classifier, param_grid, cv=5, n_iter=50, scoring='accuracy') # Of: scoring='roc_auc'
grid_search.fit(X, y)

print(grid_search.best_params_)
print(grid_search.best_score_)



# TODO move this to a separate cell in a notebook
# Plot the ROC-curve
y_score = classifier.predict_proba(X_test)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_score[:,1])

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Compute the AUC-ROC score of the logistic regression model
auc_score = sklearn.metrics.roc_auc_score(y_test, y_score[:,1])
print(f"AUC score: {auc_score}")
