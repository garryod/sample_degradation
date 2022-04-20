from typing import Any, Tuple, TypeVar

from h5py import Dataset
from nexusformat.nexus import NXFile
from numpy import dtype, ndarray, number, s_, zeros_like

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


class MappedReader:
    """Open a mapped dataset and provide an interator of frames."""

    def __init__(self, path: str, dataset: str, frame_dims: int = 2):
        self.frame_dims = frame_dims
        self.dataset: Dataset = NXFile(path)[dataset]

    def __iter__(self) -> "MappedReader":
        self._index = zeros_like(self.dataset.shape[: -self.frame_dims])
        return self

    def __next__(self) -> ndarray[Tuple[int, ...], dtype[number]]:
        frame = self[tuple(self._index)]
        self._increment_index()
        return frame

    def __getitem__(
        self, index: Tuple[int, ...]
    ) -> ndarray[Tuple[int, ...], dtype[number]]:
        return self.dataset[index]

    def _increment_index(self):
        self._index[-1] += 1
        for pos in reversed(range(len(self._index))):
            if self._index[pos] == self.dataset.shape[pos]:
                if pos == 0:
                    raise StopIteration
                self._index[pos - 1] += 1
                self._index[pos] = 0
