from itertools import chain
from math import prod
from typing import Generator, Iterable, NamedTuple, Tuple, Union, cast

from h5py import Dataset as H5Dataset
from nexusformat.nexus import NXFile
from numpy import dtype, ndarray, number
from torch.utils.data import Dataset

from sample_degradation.utils.hdf5_filters import load_hdf5_filters


class DatasetRef(NamedTuple):
    """An immutable reference to a hdf5 dataset by file name & dataset key."""

    file: str
    key: str


class FrameRef(NamedTuple):
    """An immutable reference to a dataset slice by file name, dataset key & slice."""

    file: str
    key: str
    slice: Tuple[Union[int, slice], ...]


def shape_slices(shape: Tuple[int, ...]) -> Generator[Tuple[int, ...], None, None]:
    """Generates possible slices for a given dataset shape in row-major order.

    Args:
        shape (Tuple[int, ...]): The shape for which slices should be generated, with
            fastest moving axis last.

    Yields:
        Generator[Tuple[int, ...], None, None]: A generator of possible slices, as
            tuples of integer indices.
    """
    for slice_idx in range(prod(shape)):
        yield tuple(
            slice_idx // prod(shape[bound_idx + 1 :]) % bound
            for bound_idx, bound in enumerate(shape)
        )


def dataset_frames(dataset_ref: DatasetRef) -> Generator[FrameRef, None, None]:
    """Generates possible frame references for a given dataset in row-major order.

    Args:
        dataset_ref (DatasetRef): An immutable reference to a hdf5 dataset by file name
            & dataset key.

    Yields:
        Generator[FrameRef, None, None]: A generator of possible frame references,
            consisting of file name, dataset key & tuple slice.
    """
    with NXFile(dataset_ref.file) as nexus_file:
        dataset_shape = cast(H5Dataset, nexus_file[dataset_ref.key]).shape
    for slice in shape_slices(dataset_shape[:-2]):
        yield FrameRef(dataset_ref.file, dataset_ref.key, slice)


class DetectorFramesDataset(Dataset):
    """A torch dataset which streams detector frames from hdf5 files."""

    def __init__(self, dataset_refs: Iterable[DatasetRef]) -> None:
        super().__init__()
        self.frame_refs = list(
            chain(*[dataset_frames(dataset) for dataset in dataset_refs])
        )
        load_hdf5_filters()

    def __len__(self) -> int:
        return len(self.frame_refs)

    def __getitem__(self, index: int) -> ndarray[Tuple[int, ...], dtype[number]]:
        frame_ref = self.frame_refs[index]
        with NXFile(frame_ref.file) as nexus_file:
            return nexus_file[frame_ref.key][frame_ref.slice]
