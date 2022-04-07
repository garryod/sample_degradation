from typing import Any, TypeVar

from nexusformat.nexus import NXFile
from numpy import array, dtype, ndarray, number

LoadedDType = TypeVar("LoadedDType", bound=dtype[number])


def load_data(path: str, key: str) -> ndarray[Any, LoadedDType]:
    """Read raw data at key from a nexus file at path.

    Args:
        path (str): The path to the nexus file.
        key (str): The key of the data within the nexus file.

    Returns:
        ndarray[Any, LoadedDType]: A numpy array of data.
    """
    with NXFile(path, "r") as file:
        return array(file[key])
