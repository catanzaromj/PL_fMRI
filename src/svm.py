"""Construct the SVM."""
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .landscapes import pad_flatten_landscape_values, select_from_list


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
    landscapes : list
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
    if len(labels) == 3:
        plC = select_from_list(landscapes, target_labels, labels[2])
        labels_ = (
            [labels[0]] * len(plA) + [labels[1]] * len(plB) + [labels[2]] * len(plC)
        )
        pls = pad_flatten_landscape_values(plA + plB + plC)
    else:
        labels_ = [labels[0]] * len(plA) + [labels[1]] * len(plB)
        pls = pad_flatten_landscape_values(plA + plB)

    svm_clf = Pipeline(
        [
            # ("Scaler", StandardScaler()),
            ("Linear SVC", LinearSVC(C=C, loss=loss, random_state=seed, dual=False)),
        ]
    )
    svm_clf.fit(pls, labels_)
    score = cross_val_score(svm_clf, pls, labels_, scoring=scoring, cv=folds)
    return score


def nontda_svm(
    list_of_vectors: list,
    labels: list,
    target_labels: list,
    seed: int = 0,
    C: int = 10,
    loss: str = "squared_hinge",
    scoring: str = "accuracy",
    folds: int = 10,
):
    """
    Construct an SVM for non-TDA processed data.

    Parameters
    ----------
    list_of_vectors : list
        DESCRIPTION.
    labels : list
        DESCRIPTION.
    target_labels : list
        DESCRIPTION.
    seed : int, optional
        DESCRIPTION. The default is 0.
    C : int, optional
        DESCRIPTION. The default is 10.
    loss : str, optional
        DESCRIPTION. The default is "squared_hinge".
    scoring : str, optional
        DESCRIPTION. The default is "accuracy".
    folds : int, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    raw_score : TYPE
        DESCRIPTION.

    """
    if seed == 0:
        seed = None

    vectors_a = select_from_list(list_of_vectors, target_labels, labels[0])
    vectors_b = select_from_list(list_of_vectors, target_labels, labels[1])

    if len(labels) == 3:
        vectors_c = select_from_list(list_of_vectors, target_labels, labels[2])
        labels_abc = (
            [labels[0]] * len(vectors_a)
            + [labels[1]] * len(vectors_b)
            + [labels[2]] * len(vectors_c)
        )
    else:
        labels_ab = [labels[0]] * len(vectors_a) + [labels[1]] * len(vectors_b)

    svm_raw_clf = Pipeline(
        [
            # ("Scaler", StandardScaler()),
            ("Linear SVC", LinearSVC(C=C, loss=loss, random_state=seed, dual=False)),
        ]
    )
    if len(labels) == 2:
        svm_raw_clf.fit(vectors_a + vectors_b, labels_ab)
        raw_score = cross_val_score(
            svm_raw_clf, vectors_a + vectors_b, labels_ab, scoring=scoring, cv=folds
        )
        return raw_score
    elif len(labels) == 3:
        svm_raw_clf.fit(vectors_a + vectors_b + vectors_c, labels_abc)
        raw_score = cross_val_score(
            svm_raw_clf,
            vectors_a + vectors_b + vectors_c,
            labels_abc,
            scoring=scoring,
            cv=folds,
        )
        return raw_score
