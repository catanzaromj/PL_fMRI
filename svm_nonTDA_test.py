from src import target_labels
from src.make_dataset import construct_vector
from src.svm import nontda_svm, landscape_svm
import logging
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)

SUPRA_LEVEL = True  # Compute the supralevel persistent homology as opposed to sub-level
# DATA_DIR = "data/"  # edit this to point to the path of the data directory.
DATA_DIR = "/run/user/1000/gvfs/smb-share:server=192.168.68.110,share=home/Sam_Vaibhav_project_files/"  # edit this to point to the path of the data directory.
SUBJECT_LIST = ["0295", "0394", "0408", "0484", "0489", "0505", "0521", "0551"]
HOMOLOGICAL_DEGREES = [0, 1]

SVM_TEST_LABELS = ["rest", "beat", "random"]


# def svm_rest_beat(subject: str, data_dir: str):
#     """

#     Parameters
#     ----------
#     subject

#     Returns
#     -------

#     """
#     vectors = construct_vector(subject=subject, data_dir=data_dir)
#     return raw_svm(
#         list_of_vectors=vectors, labels=["rest", "beat"], target_labels=target_labels
#     )


# def svm_random_beat(subject: str, data_dir: str):
#     """

#     Parameters
#     ----------
#     subject

#     Returns
#     -------

#     """
#     vectors = construct_vector(subject=subject, data_dir=data_dir)
#     return raw_svm(
#         list_of_vectors=vectors, labels=["random", "beat"], target_labels=target_labels
#     )


# def svm_rest_random(subject: str, data_dir: str):
#     """

#     Parameters
#     ----------
#     subject

#     Returns
#     -------

#     """
#     vectors = construct_vector(subject=subject, data_dir=data_dir)
#     return raw_svm(
#         list_of_vectors=vectors, labels=["rest", "random"], target_labels=target_labels
#     )


# def svm_all_three(subject: str, data_dir: str):
#     """

#     Parameters
#     ----------
#     subject

#     Returns
#     -------

#     """
#     vectors = construct_vector(subject=subject, data_dir=data_dir)
#     return raw_svm(
#         list_of_vectors=vectors,
#         labels=["rest", "random", "beat"],
#         target_labels=target_labels,
#     )


def svm_raw_list(subject: str, data_dir: str):
    logging.info(f"Running on svm_raw_list on {subject}")
    return_list = []
    logging.info("Running construct_vector")
    vectors = construct_vector(subject=subject, data_dir=data_dir)
    return_list.append(
        nontda_svm(
            list_of_vectors=vectors,
            labels=["rest", "beat"],
            target_labels=target_labels,
        ).mean()
    )
    logging.info("Completed rest vs beat")
    return_list.append(
        nontda_svm(
            list_of_vectors=vectors,
            labels=["random", "beat"],
            target_labels=target_labels,
        ).mean()
    )
    logging.info("Completed random vs beat")
    return_list.append(
        nontda_svm(
            list_of_vectors=vectors,
            labels=["rest", "random"],
            target_labels=target_labels,
        ).mean()
    )
    logging.info("Completed rest vs random")
    return_list.append(
        nontda_svm(
            list_of_vectors=vectors,
            labels=["rest", "random", "beat"],
            target_labels=target_labels,
        ).mean()
    )
    logging.info("Completed all three")
    return return_list


def svm_raw_pd_test(subject_list: str = SUBJECT_LIST, data_dir: str = DATA_DIR):
    avg_accur = pd.DataFrame(
        columns=["Rest vs Beat", "Random vs Beat", "Rest vs Random", "All Three"]
    )
    for subject in subject_list:
        avg_accur.loc[subject] = svm_raw_list(subject, data_dir=data_dir)
    return avg_accur


avg_accur = svm_raw_pd_test(subject_list=SUBJECT_LIST)
ax = sns.heatmap(avg_accur, annot=True, linewidths=0.5, cmap="YlOrRd_r")
ax.set_title("Non-TDA SVM classification accuracies as a function of pairing, unscaled")
ax.set_xticklabels(avg_accur.columns)


# def svm_landscape_list(subject: str):
#     return_list = []
#     return_list.append(landscape_svm(subject=subject, data_dir=DATA_DIR).mean())
#     logging.info("Completed rest vs beat")
#     return_list.append(svm_random_beat(subject=subject, data_dir=DATA_DIR).mean())
#     logging.info("Completed random vs beat")
#     return_list.append(svm_rest_random(subject=subject, data_dir=DATA_DIR).mean())
#     logging.info("Completed rest vs random")
#     return_list.append(svm_all_three(subject=subject, data_dir=DATA_DIR).mean())
#     logging.info("Completed all three")
#     return return_list
