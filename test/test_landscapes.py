import numpy as np
import pytest
from persim.landscapes import PersLandscapeApprox

from src.landscapes import pad_flatten_landscape_values, select_from_list


class TestLandscapes:
    def test_select_from_list(self):
        with pytest.raises(ValueError):
            select_from_list(
                landscapes=["1", "2"], list_of_labels=["0"], target_label=["0"]
            )
        assert ["a", "c", "e"] == select_from_list(
            landscapes=["a", "b", "c", "d", "e"],
            list_of_labels=["0", "1", "0", "1", "0"],
            target_label="0",
        )

    def test_pad_flatten(self):
        P = PersLandscapeApprox(
            start=0,
            stop=5,
            num_steps=6,
            values=np.array([[0, 1, 2, 2, 1, 0], [0, 0, 1, 0, 0, 0]]),
        )
        Q = PersLandscapeApprox(
            start=0,
            stop=4,
            num_steps=4,
            values=np.array([[0, 1, 1, 0]]),
        )
        np.testing.assert_array_equal(
            pad_flatten_landscape_values([P, Q]),
            [
                np.array([0, 1, 2, 2, 1, 0, 0, 0, 1, 0, 0, 0]),
                np.array([0, 0.75, 1, 0.75, 0, 0, 0, 0, 0, 0, 0, 0]),
            ],
        )
