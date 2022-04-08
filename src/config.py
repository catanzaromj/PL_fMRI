"""Constants used in the calculations.

target_labels: The true labelling of time slices, used for classification.

total_time: Number of time slices in the acquisition.
"""


target_labels = (
    ["rest"] * 3
    + ["beat"] * 13
    + ["rest"] * 8
    + ["random"] * 15
    + ["rest"] * 6
    + ["beat"] * 15
    + ["rest"] * 5
    + ["random"] * 15
    + ["rest"] * 26
    + ["beat"] * 15
    + ["rest"] * 5
    + ["random"] * 15
    + ["rest"] * 6
    + ["random"] * 15
    + ["rest"] * 26
    + ["beat"] * 15
    + ["rest"] * 7
)

total_time = len(target_labels)
