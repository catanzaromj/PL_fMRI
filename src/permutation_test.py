"""Construct the permutation test."""
import random
from persim.landscapes import average_approx, snap_pl
from .landscapes import select_from_list


def permutation_test(
    landscapes: list, labels: list[str], num_perms: int = 1500, seed: int = 42
):
    """
    Compute the permutation test of landscapes with labellings.

    The permutation test seeks to determine if the labelling of landscapes by
    task modality is significant. To establish a baseline, first compute the
    averages of the landscapes with the original labelling, take their difference,
    and compute the supremum norm of this difference. This number is the baseline
    significance. Now shuffle the labellings, compute the new
    average landscapes with respect to the shuffled labels, compute their
    difference and finally its supremum norm. Compare this number to the
    baseline significance. If it is greater, than we say the initial labelling
    was more signifcant; if not, then it is not. Repeat this shuffling
    `num_perms` times. The returned p-val is the number of non-significant
    labellings divided by `num_perms`.

    Parameters
    ----------
    landscapes: list
        List of landscapes to perform the permutation test on.
    labels: list
        List of two strings chosen from "rest", "beat", or "random" to use in
        performing the permutation test.
    num_perms : int
        Number of shuffles used in the permutation test.
    seed: int, optional
        Random seed for consistency among repeated runs.

    Returns
    -------
    The p-value of the test.

    """
    from src import target_labels

    if seed:
        random.seed(seed)

    plA = select_from_list(landscapes, target_labels, labels[0])
    plB = select_from_list(landscapes, target_labels, labels[1])
    avg_A = average_approx(plA)
    avg_B = average_approx(plB)
    [avg_A_snapped, avg_B_snapped] = snap_pl([avg_A, avg_B])
    true_diff_pl = avg_A_snapped - avg_B_snapped
    significance = true_diff_pl.sup_norm()

    combined_pls = plA + plB
    sig_count = 0

    for shuffle in range(num_perms):
        A_indices = random.sample(range(len(combined_pls)), (len(plA) + len(plB)) // 2)
        B_indices = [_ for _ in range(len(combined_pls)) if _ not in A_indices]

        A_pl = [combined_pls[_] for _ in A_indices]
        B_pl = [combined_pls[_] for _ in B_indices]

        A_avg = average_approx(A_pl)
        B_avg = average_approx(B_pl)
        [A_avg_sn, B_avg_sn] = snap_pl([A_avg, B_avg])

        shuff_diff = A_avg_sn - B_avg_sn
        if shuff_diff.sup_norm() >= significance:
            sig_count += 1

    p_val = sig_count / num_perms
    return p_val
