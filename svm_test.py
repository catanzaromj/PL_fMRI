import logging
import pandas as pd
import seaborn as sns

from src import target_labels as TARGET_LABELS
from src.landscapes import construct_landscapes
from src.svm import landscape_svm

SUPRA_LEVEL = True  # Compute the supralevel persistent homology as opposed to sub-level
DATA_DIR = "/run/user/1000/gvfs/smb-share:server=192.168.68.110,share=home/Sam_Vaibhav_project_files/"  # edit this to point to the path of the data directory"  # edit this to point to the path of the data directory.
SUBJECT_LIST = ["0295", "0394", "0408", "0484", "0489", "0505", "0521", "0551"]
HOMOLOGICAL_DEGREES = [0, 1]

PERM_TEST_LABELS = ["rest", "beat"]
SVM_TEST_LABELS = ["rest", "beat"]


logging.basicConfig(level=logging.INFO)

avg_accur = pd.DataFrame(
    columns=["Rest vs Beat", "Random vs Beat", "Rest vs Random", "All Three"]
)


for subject in SUBJECT_LIST:
    logging.info(f"Beginning subject {subject}")
    pl_list = []
    for hom_deg in HOMOLOGICAL_DEGREES:
        logging.info(f"Beginning homological degree {hom_deg}")
        pl = construct_landscapes(subject=subject, hom_deg=hom_deg, data_dir=DATA_DIR)
        pl_list.append(pl)
        logging.info(f"Ending hom deg {hom_deg}")
    svm_vals = []
    logging.info("Beginning rest vs beat")
    svm_vals.append(
        landscape_svm(
            landscapes=pl_list[0] + pl_list[1],
            labels=["rest", "beat"],
            target_labels=TARGET_LABELS * 2,
        ).mean()
    )
    logging.info("finishing rest vs beat")
    logging.info("beginning random vs beat")
    svm_vals.append(
        landscape_svm(
            landscapes=pl_list[0] + pl_list[1],
            labels=["random", "beat"],
            target_labels=TARGET_LABELS * 2,
        ).mean()
    )
    logging.info("finishing random vs beat")
    logging.info("beginning rest vs random")
    svm_vals.append(
        landscape_svm(
            landscapes=pl_list[0] + pl_list[1],
            labels=["rest", "random"],
            target_labels=TARGET_LABELS * 2,
        ).mean()
    )
    logging.info("finishing rest vs random")
    logging.info("beginning all three")
    svm_vals.append(
        landscape_svm(
            landscapes=pl_list[0] + pl_list[1],
            labels=["rest", "beat", "random"],
            target_labels=TARGET_LABELS * 2,
        ).mean()
    )
    logging.info("ending all three")
    avg_accur.loc[subject] = svm_vals
ax = sns.heatmap(avg_accur, annot=True, linewidths=0.5, cmap="YlOrRd_r")
ax.set_title("TDA-based SVM classification accuracies as a function of pairing")
ax.set_xticklabels(avg_accur.columns)
