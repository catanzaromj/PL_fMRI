
from src import target_labels
from src.make_dataset import construct_vector
from src.svm import raw_svm
import logging
logging.basicConfig(level=logging.INFO)

SUPRA_LEVEL = True  # Compute the supralevel persistent homology as opposed to sub-level
# DATA_DIR = "data/"  # edit this to point to the path of the data directory.
DATA_DIR = "/run/user/1000/gvfs/smb-share:server=192.168.68.110,share=home/Sam_Vaibhav_project_files/"  # edit this to point to the path of the data directory.
SUBJECT_LIST = ["0295","0394","0408","0484","0489","0505","0508", "0521","0551"]
HOMOLOGICAL_DEGREES = [0, 1]

SVM_TEST_LABELS = ["rest", "beat", "random"]

def svm_rest_beat(subject: str, hom_deg: int, labels: list):
    """

    Parameters
    ----------
    subject
    hom_deg
    labels

    Returns
    -------

    """
    vectors = construct_vector(subject=subject,hom_deg=hom_deg,data_dir=DATA_DIR)
    return raw_svm(list_of_vectors=vectors, labels=labels,target_labels=target_labels)

def svm_random_beat(subject: str, hom_deg: int, labels: list):
    """

    Parameters
    ----------
    subject
    hom_deg
    labels

    Returns
    -------

    """
    vectors = construct_vector(subject=subject,hom_deg=hom_deg,data_dir=DATA_DIR)
    return raw_svm(list_of_vectors=vectors, labels=labels,target_labels=target_labels)

def svm_rest_random(subject: str, hom_deg: int, labels: list):
    """

    Parameters
    ----------
    subject
    hom_deg
    labels

    Returns
    -------

    """
    vectors = construct_vector(subject=subject,hom_deg=hom_deg,data_dir=DATA_DIR)
    return raw_svm(list_of_vectors=vectors, labels=labels,target_labels=target_labels)

def svm_raw_test(subject:str):
    logging.info(f'Running on svm_raw_test on {subject}')
    write_file = open('svm_raw_results.txt', 'w')
    write_file.write(f'Running on subject {subject}\n')
    return_dict = {}
    for hom_deg in HOMOLOGICAL_DEGREES:
        logging.info(f'Currently running in homological degree {hom_deg}')
        write_file.write(f'Running on homological degree {hom_deg}\n')
        return_dict[hom_deg] = {}
        return_dict[hom_deg]['rest_vs_beat'] = svm_rest_beat(subject=subject,hom_deg=hom_deg,labels=['rest', 'beat'])
        logging.info('Completed rest vs beat')
        write_file.write(f'Rest vs Beat: {return_dict[hom_deg]["rest_vs_beat"]}\n')
        return_dict[hom_deg]['random_vs_beat'] = svm_rest_beat(subject=subject,hom_deg=hom_deg,labels=['random', 'beat'])
        logging.info('Completed random vs beat')
        write_file.write(f'Random vs Beat: {return_dict[hom_deg]["random_vs_beat"]}\n')
        return_dict[hom_deg]['rest_vs_random'] = svm_rest_beat(subject=subject, hom_deg=hom_deg, labels=['rest', 'random'])
        logging.info('Completed rest vs random')
        write_file.write(f'Rest vs Random: {return_dict[hom_deg]["rest_vs_random"]}\n')
    write_file.close()
    return return_dict

raw_results = []
for subject in SUBJECT_LIST:
    logging.info(f'Running on subject {subject}')
    raw_results.append(svm_raw_test(subject=subject))

