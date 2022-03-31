from typing import TypeVar

from numpy import bool_, broadcast_to, number
from numpy.ma import masked_where
from numpy.typing import NDArray

FrameDType = TypeVar("FrameDType", bound=number)


def mask_frames(
    frames: NDArray[FrameDType],
    mask: NDArray[bool_],
) -> NDArray[FrameDType]:
    """Replaces masked elemenets of frames in a stack with zero.

    Args:
        frames (NDArray[FrameDType]): A stack of frames to be masked.
        mask (NDArray[bool_]): The boolean mask to apply to each frame.

    Returns:
        NDArray[FrameDType]: A stack of frames where masked elements are
            replaced by zero.
    """
    return masked_where(broadcast_to(mask, frames.shape), frames).filled(fill_value=0)
