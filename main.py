"""Main script for applying TDA to fMRI for task modality detection.

The main steps are as follows.
1. Pre-process the data: convert matlab files to plain text files, apply the
   mask, and convert the files into perseus input files.
2. Run perseus on the input files.
3. Convert the perseus output into a numpy.ndarray to be fed into persim.
4. Compute the Persistence Landscapes of each time slice.
5. Label and process (pad) the landscapes for the statistical tests.
6. Apply one of two statistical tests:
    6a. The permutation test. Given two modality labels, compute the average PL
    of each label and then take the sup norm of their difference.
    Shuffle the labellings, compute new averages and the new sup norm difference.
    Compare this to the original differnce and determine if the shuffled labelling
    is significant.
    6b. SVM. Given two modality labels, construct a linear SVM for each label
    and validate it using 10-fold cross validation.

The first two steps are omitted because the raw data files are too large. Thus,
the workflow begins with reading from perseus input files which have been provided
in 'data/postprocessed'.
"""
from persim.landscapes import PersLandscapeApprox

from src.landscapes import (
    pad_flatten_landscape_values,
    construct_landscapes,
    select_from_list,
)
from src.permutation_test import permutation_test
from src.svm import landscape_svm
from src import target_labels

SUPRA_LEVEL = True  # Compute the supralevel persistent homology as opposed to sub-level
DATA_DIR = "data/"  # edit this to point to the path of the data directory.
SUBJECT_LIST = ["0508"]
HOMOLOGICAL_DEGREES = [0, 1]

PERM_TEST_LABELS = ["rest", "beat"]
SVM_TEST_LABELS = ["rest", "beat"]

### For the code review: The raw matlab files are very large, so I haven't
### include them in this repository. If they were included,
### the following three lines would
### compute the perseus input files from them and use gudhi/perseus to compute
### the persistence.
# from src.make_dataset import construct_perseus_input_files
# construct_perseus_input_files(SUBJECT_LIST, supra=SUPRA_LEVEL)
# run perseus on those files

pl_list = []
for subject in SUBJECT_LIST:
    for hom_deg in HOMOLOGICAL_DEGREES:
        pl_list.append(
            construct_landscapes(subject=subject, hom_deg=hom_deg, data_dir=DATA_DIR)
        )

# Permutation Test; example call
p_val0 = permutation_test(pl_list[0], PERM_TEST_LABELS)
p_val1 = permutation_test(pl_list[1], PERM_TEST_LABELS)


## SVM
# Build a classifier using H0
svm_random_rest_0, score_random_rest_0 = landscape_svm(
    landscapes=pl_list[0],
    labels=SVM_TEST_LABELS,
    target_labels=target_labels,
    folds=10,
)

# Build a classifier using H1
svm_random_rest_1, score_random_rest_1 = landscape_svm(
    landscapes=pl_list[1],
    labels=SVM_TEST_LABELS,
    target_labels=target_labels,
    folds=10,
)

# # Build a classifier using H2
# svm_random_rest_2, score_random_rest_2 = landscape_svm(
#     landscapes=pl_list[2],
#     labels=SVM_TEST_LABELS,
#     target_labels=target_labels,
#     folds=10,
# )

# Build a classifier using both H0 and H1
svm_random_rest_01, score_random_rest_01 = landscape_svm(
    landscapes=pl_list[0] + pl_list[1],
    labels=SVM_TEST_LABELS,
    target_labels=target_labels * 2,
    folds=10,
)

# Build a classifier by truncating each of H0 and H1 first
truncated_pl0 = [
    PersLandscapeApprox(
        start=pl_list[0][i].start,
        stop=pl_list[0][i].stop,
        num_steps=pl_list[0][i].num_steps,
        hom_deg=pl_list[0][i].hom_deg,
        values=pl_list[0][i][:5],
    )
    for i in range(len(pl_list[0]))
]
truncated_pl1 = [
    PersLandscapeApprox(
        start=pl_list[1][i].start,
        stop=pl_list[1][i].stop,
        num_steps=pl_list[1][i].num_steps,
        hom_deg=pl_list[1][i].hom_deg,
        values=pl_list[1][i][:5],
    )
    for i in range(len(pl_list[1]))
]
svm_random_rest_trunc_01, score_random_rest_trunc_01 = landscape_svm(
    landscapes=truncated_pl0 + truncated_pl1,
    labels=SVM_TEST_LABELS,
    target_labels=target_labels * 2,
    folds=10,
    C=100,
)
