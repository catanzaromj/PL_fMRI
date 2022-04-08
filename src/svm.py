"""Construct the SVM."""
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .landscapes import select_from_list, pad_flatten_landscape_values


def landscape_svm(
    landscapes: list,
    labels: list,
    target_labels: list,
    seed: int = 0,
    C: int = 10,
    loss: str = "squared_hinge",
    scoring: str = "accuracy",
    folds: int = 10,
):
    """
    Construct an SVM pipeline with standard scaling.

    Parameters
    ----------
    landscapess : list
        List of landscapes.

    labels: list
        List of two strings chosen from "rest", "beat", or "random" to use in
        performing the permutation test.

    target_labels: list
        List of target labels for the classifier. Must be same length as landscapes.

    C: int
        Hyperparameter for the linear SVM.

    loss: str
        The loss function for the linear SVM.

    seed: int
        Random seed for repeated runs.

    folds: int
        Number of folds for cross-validation

    Returns
    -------
    A sklearn-style pipeline.

    """
    if seed == 0:
        seed = None
    plA = select_from_list(landscapes, target_labels, labels[0])
    plB = select_from_list(landscapes, target_labels, labels[1])
    pls = pad_flatten_landscape_values(plA + plB)
    labels_AB = [labels[0]] * len(plA) + [labels[1]] * len(plB)

    svm_clf = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("Linear SVC", LinearSVC(C=C, loss=loss, random_state=seed, dual=False)),
        ]
    )
    svm_clf.fit(pls, labels_AB)
    score = cross_val_score(svm_clf, pls, labels_AB, scoring=scoring, cv=folds)
    return svm_clf, score
