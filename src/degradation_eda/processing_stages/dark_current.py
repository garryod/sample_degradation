from typing import Any, TypeVar

from numpy import array
from numpy.ma import MaskedArray, masked_array

from degradation_eda.utils.uncertain_maths import (
    Uncertain,
    subtract_uncertain,
    uncertain,
)

FrameShape = TypeVar("FrameShape", bound=Any)


def correct_dark_current(
    frames: MaskedArray[FrameShape, Uncertain],
    dark_current: float,
) -> MaskedArray[FrameShape, Uncertain]:
    """Correct for dark current by subtracting a constant.

    Args:
        frames (MaskedArray[FrameShape, Uncertain]): A stack of frames to be corrected
            and their uncertaintities.
        dark_current (float): The dark current flux.

    Returns:
        MaskedArray[FrameShape, Uncertain]: The corrected stack of frames and their
            updated uncertainties.
    """
    return masked_array(
        subtract_uncertain(frames.data, array([(dark_current, 0.0)], dtype=uncertain)),
        frames.mask,
    )
