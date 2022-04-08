"""Compute persistence landscapes from perseus files and auxiliary functions."""

import os
import numpy as np
from persim.landscapes import PersLandscapeApprox, snap_pl


def perseus_to_sktda(
    subject: str, hom_deg: int, time: int, data_dir: str
) -> np.ndarray:
    """
    Convert perseus output to scikit-tda-style persistence diagrams.

    Parameters
    ----------
    subject : str
        The subject number to be analyed.
    hom_deg : int
        The homological degree.
    time : int
        The time slice.
    data_dir : str
        The path to the data directory. This should contain the raw, preprocessed,
        and postprocessed directories.

    Returns
    -------
    A numpy.ndarray of the persistence diagram.

    """
    subject_prs_path = os.path.join(data_dir, "postprocessed", subject)
    subject_pd = []
    subject_prs = os.path.join(
        subject_prs_path,
        "patient_"
        + subject
        + "_time_"
        + str(time)
        + "_output_"
        + str(hom_deg)
        + ".txt",
    )
    with open(subject_prs, "r") as prs_file:
        for line in prs_file.readlines():
            x, y = line.strip().split(" ")
            if y != "-1":
                subject_pd.append(np.array([int(x), int(y)]))
            else:
                subject_pd.append(np.array([int(x), np.inf]))
    return (
        [np.array([])] * hom_deg
        + [np.array(subject_pd)]
        + [np.array([])] * (3 - hom_deg)
    )


def construct_landscapes(subject: str, hom_deg: int, data_dir: str) -> list:
    """
    Construct the list of persistence landscapes.

    Given a subject and a homological degree, construct the persistence
    landscapes for that subject in that homological degree for all time slices.

    Parameters
    ----------
    subject : int
        The subject number to be analyzed.
    hom_deg : int
        The homological degree.
    data_dir : str
        The path to the data directory.

    Returns
    -------
    List of landscapes

    """
    pl_list = []
    for time in range(210):
        diagrams = perseus_to_sktda(
            subject=subject, hom_deg=hom_deg, time=time, data_dir=data_dir
        )
        pl_list.append(
            PersLandscapeApprox(dgms=diagrams, hom_deg=hom_deg, num_steps=1800)
        )
    return pl_list


def select_from_list(landscapes: list, list_of_labels: list, target_label: str) -> list:
    """
    Select a sublist of landscapes based on label.

    Given a list `landscapes` and a total list of labels `list_of_labels`,
    return those entries of `landscapes` whose corresponding entry in
    `list_of_labels` matches `target_label`.

    Parameters
    ----------
    landscapes: list
        A list to be picked from.
    list_of_labels: list
        A complete labelling of `landscapes`.
    target_label: str
        The desired label type to be selected.

    Returns
    -------
    A list of landscapes with label equal to `target_label`.
    """
    if len(landscapes) != len(list_of_labels):
        raise ValueError("landscapes and list_of_labels must be the same length")

    pl_list = []
    for idx, modality in enumerate(list_of_labels):
        if modality == target_label:
            pl_list.append(landscapes[idx])
    return pl_list


def pad_flatten_landscape_values(landscapes: list) -> list:
    """
    Add zeroes to landscape values so they are all the same length and flatten them.

    The list `landscapes` may contain landscapes of different depths and
    therefore, will be vectors of different length in Euclidean space. This
    method pads them to all be of the same length (the max length). This results
    in a list of numpy arrays of size (max_depth, num_steps), which then need
    to be flattened to produce a vector of length max_depth * num_steps.

    NOTE:: Does not pad in place. Returns a list of values rather than a list
    of landscapes.

    Parameters
    ----------
    landscapes : list
        A list of landscapes

    Returns
    -------
    The padded and flattened landscape values.

    """
    landscapes = snap_pl(landscapes)

    max_depth = np.max([landscape.max_depth for landscape in landscapes])
    pl_values = [landscape.values for landscape in landscapes]
    num_steps = landscapes[0].num_steps

    for idx, value in enumerate(pl_values):
        if np.shape(value)[0] == max_depth:
            pl_values[idx] = value.flatten()
            continue
        else:
            pl_values[idx] = np.append(
                value, [np.array([0] * num_steps * (max_depth - np.shape(value)[0]))]
            )
    return pl_values
