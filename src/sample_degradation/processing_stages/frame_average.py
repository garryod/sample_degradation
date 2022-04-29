from typing import Any, TypeVar

from numpy import array
from numpy.ma import MaskedArray, masked_array

from sample_degradation.utils.uncertain_maths import (
    Uncertain,
    divide_uncertain,
    sum_uncertain,
    uncertain,
)

FrameShape = TypeVar("FrameShape", bound=Any)


def average_frames(
    frames: MaskedArray[FrameShape, Uncertain]
) -> MaskedArray[FrameShape, Uncertain]:
    """Average all frames over the leading axis.

    Args:
        frames (MaskedArray[FrameShape, Uncertain]): A stack of frames to be averaged
            and their uncertainities.

    Returns:
        MaskedArray[FrameShape, Uncertain]: A frame containing the average pixel values
            of all frames in the stack and their uncertainties.
    """
    return masked_array(
        divide_uncertain(
            sum_uncertain(frames, axis=0),
            array([(frames.shape[0], 0.0)], dtype=uncertain),
        ),
        mask=frames.mask[0],
    )
