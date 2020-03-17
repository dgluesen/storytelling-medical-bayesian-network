import numpy as np
from medical_storytelling.datasets import Oasis, Cancer


def test_oasis():
    """
    Tests the Oasis dataset class for functionality.
    """
    try:
        test_data = Oasis()
        assert test_data.features_arr.shape == test_data.features_df.shape
        assert isinstance(test_data[0], dict)
    except FileNotFoundError:
        raise FileNotFoundError(
            "File was not found! Make sure the data csv file is in `repository/dat/alzheimer/`. If you're using a custom path, ignore this error and add the path to this test function!"
        )

def test_cancer():
    """
    Tests the Cancer dataset class for functionality.
    """
    try:
        test_data = Cancer()
        assert test_data.features_arr.shape == test_data.features_df.shape
        assert isinstance(test_data[0], dict)
    except FileNotFoundError:
        raise FileNotFoundError(
            "File was not found! Make sure the data csv file is in `repository/dat/alzheimer/`. If you're using a custom path, ignore this error and add the path to this test function!"
        )
