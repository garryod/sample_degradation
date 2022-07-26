from itertools import tee
from math import prod
from typing import Any, Generator, Iterable, Iterator, Tuple, TypeVar

import hdf5plugin  # noqa: F401
from h5py import Dataset
from nexusformat.nexus import NXFile
from numpy import array, dtype, ndarray, number, s_

T = TypeVar("T")
LoadedDType = TypeVar("LoadedDType", bound=dtype[number])


def _repeat_outer(items: Iterable[T], repeats: int) -> Generator[T, None, None]:
    for copy in tee(items, repeats):
        for item in copy:
            yield item


def _repeat_inner(items: Iterable[T], repeats: int) -> Generator[T, None, None]:
    for item in items:
        for _ in range(repeats):
            yield item


def _multidimensional_indices(shape: Tuple[int, ...]) -> Iterable[Tuple[int, ...]]:
    return zip(
        *[
            _repeat_outer(
                list(_repeat_inner(range(shape[idx]), prod(shape[idx + 1 :]))),
                prod(shape[:idx]),
            )
            for idx in range(len(shape))
        ]
    )


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
        dataset: Dataset = file[key]
        if dataset.shape == ():
            if use_slice != s_[:]:
                raise ValueError("Cannot take slice of scalar dataset")
            return array(dataset)
        else:
            return dataset[use_slice]


def map_frames(
    path: str, key: str, frame_dims: int = 2
) -> Iterator[ndarray[Tuple[int, ...], dtype[number]]]:
    """Generate frames from a mapped dataset, in row-major order.

    Args:
        path (str): The path to the nexus file.
        key (str): The key of the data within the nexus file.
        frame_dims (int, optional): The number of dimensions occupied by a frame.
            Defaults to 2.

    Yields:
        Iterator[ndarray[Tuple[int, ...], dtype[number]]]: A interator of loaded frames.
    """
    dataset: Dataset = NXFile(path)[key]
    frame_indices = _multidimensional_indices(dataset.shape[:-frame_dims])

    return map(lambda frame_indices: dataset[frame_indices], frame_indices)
