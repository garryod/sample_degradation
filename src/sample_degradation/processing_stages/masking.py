from typing import Any, TypeVar

from numpy import bool_, broadcast_to, dtype, ndarray, number
from numpy.ma import MaskedArray, masked_where

FrameDType = TypeVar("FrameDType", bound=dtype[number])
FramesShape = TypeVar("FramesShape", bound=Any)


def mask_frames(
    frames: ndarray[FramesShape, FrameDType],
    mask: ndarray[Any, dtype[bool_]],
) -> MaskedArray[FramesShape, FrameDType]:
    """Replaces masked elemenets of frames in a stack with zero.

    Args:
        frames (ndarray[FrameShape, FrameDType]): A stack of frames to be masked.
        mask (ndarray[Any, dtype[bool_]]): The boolean mask to apply to each frame.

    Returns:
        ndarray[FrameShape, FrameDType]: A stack of frames where pixels
    """
    return masked_where(broadcast_to(mask, frames.shape), frames)
