from typing import TypeVar

from nexusformat.nexus import NXFile
from numpy import array, number
from numpy.typing import NDArray

LoadedDType = TypeVar("LoadedDType", bound=number)


def load_data(path: str, key: str) -> NDArray[LoadedDType]:
    """Read raw data at key from a nexus file at path.

    Args:
        path (str): The path to the nexus file.
        key (str): The key of the data within the nexus file.

    Returns:
        NDArray: A numpy array of data.
    """
    with NXFile(path, "r") as file:
        return array(file[key])
