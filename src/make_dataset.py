"""Transform the raw matlab input files into a numpy arrays of persistence diagrams.

The raw matlab files contain signal amplitudes for the entire acquisition, not
just the ACC. The mask filters these other regions out, leaving the ACC.

"""

import os

import gudhi
import h5py as h5
import numpy as np


def apply_mask(subject: str, mask: np.ndarray, data_dir: str, supra: bool) -> None:
    """
    Apply the mask to the subject and convert to a perseus input file.

    This method reads from the 'raw' subdirectory of `data_dir` and writes
    to the 'preprocessed' subdirectory of `data_dir`. It does not return
    anything.

    This method should not be called directly; use `construct_dataset` to
    avoid reloading the (potentially large) mask.

    Parameters
    ----------
    subject : str
        The subject number to be parsed.

    mask: np.ndarray
        The mask to be applied to subject.

    data_dir : str
        The path to the data directory.

    supra : bool
        True if supralevelset persistent homology is to be computed.

    Returns
    -------
    None.
    """
    subject_data_path = os.path.join(
        data_dir, "raw", subject, "rocd" + subject + ".mat"
    )
    subject_data = np.array(h5.File(subject_data_path, "r"))
    max_fmri = int(np.round(np.amax(subject_data)))

    from src import total_time

    for time in range(total_time):
        prs_filename = "patient_" + subject + "_time_" + str(time) + ".prs"
        prs_file = open(
            os.path.join(data_dir, "preprocessed", subject, prs_filename), "w"
        )
        prs_file.write("3/n")  # Write dimension to top of perseus input file

        for z in range(35, 85):
            for x in range(40, 77):
                for y in range(83, 123):
                    if int(np.round(mask[z, x, y])) != 0:
                        if supra:
                            prs_file.write(
                                f"{z} {x} {y} {np.round(max_fmri - subject_data[time,z,x,y])}"
                            )
                        else:
                            prs_file.write(
                                f"{z} {x} {y} {np.round(subject_data[time,z,x,y])}"
                            )
        prs_file.close()


def construct_perseus_input_files(subjects: str, data_dir: str, supra: bool) -> None:
    """
    Load the mask and apply it to each subject.

    NOTE:: Asadur accidentally hardcoded the mask as 'ocd0408', so use this
    number for all subjects, not just subject 0408.

    Parameters
    ----------
    subjects : str | list(str)
        A (list of) subject number(s) to be analyzed.

    data_dir : str
        The path to the data directory.

    supra : bool
        True if supralevelset persistent homology is to be computed.

    Returns
    -------
    None.

    """
    if type(subjects) is str:
        subjects = [subjects]
    mask_path = data_dir + "rDACC.mat"
    mask = h5.File(mask_path, "r")
    mask = np.array(mask.get("ocd0408"))  # 0408 is used for all subject runs.

    for subject in subjects:
        apply_mask(subject, mask, data_dir, supra)


def construct_diagrams_gudhi(
    subject: str, hom_deg: int, time: int, data_dir: str
) -> np.ndarray:
    """
    Construct persistence diagrams directly from gudhi.

    Parameters
    ----------
    subject : str
        A subject number to be analyzed.

    hom_deg : int
        The homological degree used to compute persistence.

    time : int
        The time slice to analyze.

    data_dir : str
        The path to the data directory.

    Returns
    -------
    A numpy.ndarray consisting of birth-death pairs.

    """
    prs_filename = "patient_" + subject + "_time_" + str(time) + ".prs"
    cubical_complex = gudhi.CubicalComplex(
        perseus_file=os.path.join(data_dir, "preprocessed", subject, prs_filename)
    )
    cubical_complex.compute_persistence(homology_coeff_field=2)
    return cubical_complex.persistence_intervals_in_dimension(hom_deg)


def construct_persistence_files(subject: str, hom_deg: int, data_dir: str) -> None:
    """
    Construct persistence diagram output files using gudhi directly.

    This method writes to the 'preprocessed' subdirectory of `data_dir`.
    It does not return anything.

    Parameters
    ----------
    subject : str
        A subject number to be analyzed.

    hom_deg : int
        The homological degree used to compute persistence.

    data_dir : str
        The path to the data directory.

    Returns
    -------
    None.

    """
    post_processing_dir = os.path.join(data_dir, "postprocessed", subject)
    from src import total_time

    for time in range(total_time):
        pd_filename = (
            "patient_"
            + subject
            + "_time_"
            + str(time)
            + "_output_"
            + str(hom_deg)
            + ".txt"
        )
        with open(os.path.join(post_processing_dir, pd_filename), "w") as pd_file:
            pds = construct_diagrams_gudhi(
                subject=subject, hom_deg=hom_deg, time=time, data_dir=data_dir
            )
            for (b, d) in pds:
                pd_file.writelines([str(b), str(d), "\n"])


def construct_vector(subject: str, data_dir: str) -> list:
    """
    Construct a vector of signal amplitude whose coordinates are ordered by the perseus input file.

    Parameters
    ----------
    subject
    data_dir

    Returns
    -------

    """
    from src import total_time

    return_list = []

    # Check if file exists. If not, create it.
    # if not os.path.exists(path=os.path.join(data_dir,"preprocessed", subject, "patient_" + subject + "_time_0.prs")):
    #    construct_persistence_files(subject=subject, hom_deg=hom_deg, data_dir=data_dir)

    subject_data_path = os.path.join(data_dir, "patient" + subject, "pers_input")
    for time in range(total_time):
        time_list = []
        prs_filename = "patient_" + subject + "_time_" + str(time) + ".prs"
        with open(os.path.join(subject_data_path, prs_filename), "r") as prs_file:
            for line in prs_file:
                if line == "3\n":
                    continue
                else:
                    time_list.append(int(line.split(" ")[-1]))
        return_list.append(time_list)
    return return_list
