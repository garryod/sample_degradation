from itertools import tee, zip_longest
from math import prod
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
)

from h5py import Dataset
from nexusformat.nexus import NXFile
from numpy import dtype, ndarray, number, s_

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
        return file[key][use_slice]


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

    return map(lambda frame_indices: dataset["dataset"][frame_indices], frame_indices)


class ProcessArgs(TypedDict, total=False):  # noqa: D101
    dataset: Dataset


_PROCESS_ARGS: ProcessArgs = {}


def _open_dataset(path: str, key: str):
    _PROCESS_ARGS["dataset"] = NXFile(path)[key]


def _read_frame(
    frame_indices: Tuple[int, ...]
) -> ndarray[Tuple[int, ...], dtype[number]]:
    return _PROCESS_ARGS["dataset"][frame_indices]


def map_frames_parallel(
    path: str,
    key: str,
    frame_dims: int = 2,
    cache_size: int = 8,
    processes: Optional[int] = None,
) -> Iterable[ndarray[Tuple[int, ...], dtype[number]]]:
    """Generate frames from a mapped dataset, in row-major order, using multiple readers.

    Creates a parallel pool to read mapped dataset frames in row-major order, acting as
    an iterator of loaded frames. With cache size and number of processes determined by

    Args:
        path (str): The path to the nexus file.
        key (str): The key of the data within the nexus file.
        frame_dims (int, optional): The number of dimensions occupied by a frame.
            Defaults to 2.
        cache_size (int, optional): The number of frames to cache, greater numbers will
            allow for more read-ahead but require more memory.
        processes (int, optional): The number of processes to use in the parallel
            pool of readers, if None then the number returned by os.cpu_count() is used.
            Defaults to None.

    Returns:
        Iterable[ndarray[Tuple[int, ...], dtype[number]]]: A interator of loaded frames.
    """
    dataset = NXFile(path)[key]
    indices = list(_multidimensional_indices(dataset.shape[:-frame_dims]))

    loaded_frames: Dict[
        Tuple[int, ...], AsyncResult[ndarray[Tuple[int, ...], dtype[number]]]
    ] = {}

    with Pool(processes, _open_dataset, (path, key)) as pool:
        for read_idx in indices[:cache_size]:
            loaded_frames[read_idx] = pool.apply_async(_read_frame, (read_idx,))
        for read_idx, yield_idx in zip_longest(indices[cache_size:], indices):
            loaded_frames[read_idx] = pool.apply_async(_read_frame, (read_idx,))
            yield loaded_frames[yield_idx].get()
            del loaded_frames[yield_idx]
