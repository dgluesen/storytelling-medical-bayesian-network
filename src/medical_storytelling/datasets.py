import pathlib, os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class Oasis(Dataset):
    def __init__(
        self,
        csv_path: str = None,
        root: str = None,
        target_column: str = "CDR",
        transform_labels: bool = True,
        normalise: str = "mean",
    ):
        """
        Basic Dataset class providing the dataset as a numpy array, pandas DataFrame, or as a torch
        Dataloader. Every attribute of this class is a pandas dataframe as default.

        This class contains the attributes `dataframe` and `array` which provide us with the full
        data as pandas.DataFrame and a numpy.ndarray respectively. 
        The attributes `features_df`, `targets_df` and `features_arr`, `targets_arr` give us the
        features and target in dataframe or array format.

        Args:
            Dataset (torch.utils.data.Dataset): Torch dataset classed for the dataloader in case class
             is used for training a pytorch neural network.
            csv_path(str, optional): Relative path to the csv file with the data.
            root (str, optional): Your root directory which is to be prepended to the csv_path. Make sure 
            this is not None is you want to provide the absolute path for the csv_file.
            target_column (str, optional): The target feature, i.e., Clinical
             Dementia Rating. Defaults to "CDR"
            transform_labels (bool, optional): If the str type categorical or ordinal columns should be transformed into integers. The mapping is saved as an attribute of the class.
            normalise (str, optional): Tells the class how the dataset is supposed to be normalised. "mean" would standardise the data with mean 0. "minmax" would normalise the data around 0 and 1. None would keep the data as is. Other values shall throw an error.
        """

        csv_file_abs = figure_out_csv_path(
            csv_path, root, "../../dat/alzheimer/oasis_longitudinal.csv"
        )

        if transform_labels:
            dataframe, self.inverse_map = cat_to_int_encoder(pd.read_csv(csv_file_abs))
        else:
            dataframe = pd.read_csv(csv_file_abs)

        if normalise == "mean":
            dataframe = (dataframe - dataframe.mean()) / dataframe.std()
        elif normalise == "minmax":
            dataframe = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
        elif normalise is None:
            pass
        else:
            raise ValueError("Unvalid value passed for argument `normalise`.")

        self.dataframe = dataframe

        self.array = dataframe.to_numpy()

        self.features_df = features_df = dataframe.loc[:, dataframe.columns != target_column]
        self.targets_df = targets_df = dataframe[target_column]

        self.features_arr = features_df.to_numpy()
        self.targets_arr = targets_df.to_numpy()

        self.columns = np.array(dataframe.columns)
        self.head = dataframe.head
        self.info = dataframe.info
        self.dtypes = dataframe.dtypes

    def __getitem__(self, idx: int):
        """
        This is method is called to get a single instance of the dataset by the pytorch dataloader.
                
        Args:
            idx (int): The index of the dataset.
        
        Returns:
            dict: An instance of the features and the target variable as an array.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        out_features = self.features_arr[idx]
        out_target = self.targets_arr[idx]
        return {"features": out_features, "target": out_target}

    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.dataframe)


class Cancer(Dataset):
    def __init__(
        self,
        csv_path: str = None,
        root: str = None,
        target_column: str = "diagnosis",
        transform_labels: bool = True,
        normalise: str = "mean",
    ):
        """
        Basic Dataset class providing the dataset as a numpy array, pandas DataFrame, or as a torch
        Dataloader. Every attribute of this class is a pandas dataframe as default.

        This class contains the attributes `dataframe` and `array` which provide us with the full
        data as pandas.DataFrame and a numpy.ndarray respectively. 
        The attributes `features_df`, `targets_df` and `features_arr`, `targets_arr` give us the
        features and target in dataframe or array format.

        Args:
            Dataset (torch.utils.data.Dataset): Torch dataset classed for the dataloader in case class
             is used for training a pytorch neural network.
            csv_path(str, optional): Relative path to the csv file with the data.
            root (str, optional): Your root directory which is to be prepended to the csv_path. Make sure 
            this is not None is you want to provide the absolute path for the csv_file.
            target_column (str, optional): The target feature, i.e., diagnosis. Defaults to "diagnosis".
            transform_labels (bool, optional): If the str type categorical or ordinal columns should be transformed into integers. The mapping is saved as an attribute of the class.
            normalise (str, optional): Tells the class how the dataset is supposed to be normalised. "mean" would standardise the data with mean 0. "minmax" would normalise the data around 0 and 1. None would keep the data as is. Other values shall throw an error.
        """

        csv_file_abs = figure_out_csv_path(csv_path, root, "../../dat/cancer/data.csv")

        if transform_labels:
            dataframe, self.inverse_map = cat_to_int_encoder(pd.read_csv(csv_file_abs))
        else:
            dataframe = pd.read_csv(csv_file_abs)

        if normalise == "mean":
            dataframe = (dataframe - dataframe.mean()) / dataframe.std()
        elif normalise == "minmax":
            dataframe = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
        elif normalise is None:
            pass
        else:
            raise ValueError("Unvalid value passed for argument `normalise`.")
        
        # the last column is empty
        self.dataframe = dataframe.iloc[:, :-1]

        self.array = dataframe.to_numpy()

        self.features_df = features_df = dataframe.loc[:, dataframe.columns != target_column]
        self.targets_df = targets_df = dataframe[target_column]

        self.features_arr = features_df.to_numpy()
        self.targets_arr = targets_df.to_numpy()

        self.columns = np.array(dataframe.columns)
        self.head = dataframe.head
        self.info = dataframe.info
        self.dtypes = dataframe.dtypes

    def __getitem__(self, idx: int):
        """
        This is method is called to get a single instance of the dataset by the pytorch dataloader.
                
        Args:
            idx (int): The index of the dataset.
        
        Returns:
            dict: An instance of the features and the target variable as an array.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        out_features = self.features_arr[idx]
        out_target = self.targets_arr[idx]
        return {"features": out_features, "target": out_target}

    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.dataframe)


#### Utilities functions ####


def cat_to_int_encoder(dataframe: pd.DataFrame):
    """
    This static method encodes every object type column of the dataframe so that it is ready
    to train with. 
            
    Args:
        dataframe (pd.DataFrame): The dataframe which should be transformed
    
    Returns:
        dataframe: The transformed dataframe
        inverse_mapping: A dictionary containing the inverse transform function for every 
        column transformed
    """
    inverse_mapping = dict()
    for col, dtype in zip(dataframe.columns, dataframe.dtypes):
        if dtype == object:
            label_encoder = LabelEncoder()
            dataframe[col] = label_encoder.fit_transform(dataframe[col])
            inverse_mapping[col] = label_encoder.inverse_transform
        else:
            continue
    return dataframe, inverse_mapping


def figure_out_csv_path(csv_path: str, root: str, relative_csv_path: str):
    """
    Method for figuring out where the csv file might be given the arguments. 
    
    Given this method, the absolute path can be provided in the argument `csv_path`, 
    relative path can be provided in `csv_path` and root in `root`. If both are not 
    provided, root will be the parent directory of this script plus 
    "../dat/alzheimer/oasis_longitudinal.csv"
                
    Args:
        csv_path (str): Argument provided for `csv_path`.
        root (str): Argument provided for `root`.
    
    Returns:
        absolute path: Our guess for the absolute path for the csv_file.
    """
    dir_path = pathlib.Path(__file__).parent.absolute()
    if root is None:
        if csv_path is None:
            csv_path = relative_csv_path
            root = dir_path
        else:
            root = "."
    csv_abs_path = os.path.join(root, csv_path)

    if not os.path.exists(csv_abs_path):
        raise FileNotFoundError(f"csv file not found at {csv_abs_path}")

    return csv_abs_path

