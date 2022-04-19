from typing import Any, TypeVar

from nexusformat.nexus import NXFile
from numpy import dtype, ndarray, number, s_

LoadedDType = TypeVar("LoadedDType", bound=dtype[number])


def load_data(
    path: str, key: str, use_slice: slice = s_[:]
) -> ndarray[Any, LoadedDType]:
    """Read raw data at key from a nexus file at path.

    Args:
        path (str): The path to the nexus file.
        key (str): The key of the data within the nexus file.
        use_slice (slice): The slice of data to load. Default all.

    Returns:
        ndarray[Any, LoadedDType]: A numpy array of data.
    """
    with NXFile(path, "r") as file:
        return file[key][use_slice]
