from typing import TypeVar

from numpy import bool_, broadcast_to, number
from numpy.ma import masked_where
from numpy.typing import NDArray

FrameDType = TypeVar("FrameDType", bound=number)


def mask_frames(
    images: NDArray[FrameDType],
    mask: NDArray[bool_],
) -> NDArray[FrameDType]:
    """Replaces masked elemenets of images in a stack with zero.

    Args:
        images (NDArray[FrameDType]): A stack of images to be masked.
        mask (NDArray[bool_]): The boolean mask to apply to each image.

    Returns:
        NDArray[FrameDType]: A stack of images where masked elements are
            replaced by zero.
    """
    return masked_where(broadcast_to(mask, images.shape), images).filled(fill_value=0)
