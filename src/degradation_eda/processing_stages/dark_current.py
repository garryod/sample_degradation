from typing import Any, TypeVar

from numpy import array, ndarray

from degradation_eda.utils.uncertain_maths import (
    Uncertain,
    subtract_uncertain,
    uncertain,
)

FrameShape = TypeVar("FrameShape", bound=Any)


def correct_dark_current(
    frames: ndarray[FrameShape, Uncertain],
    dark_current: float,
) -> ndarray[FrameShape, Uncertain]:
    """Correct for dark current by subtracting a constant.

    Args:
        frames (ndarray[FrameShape, Uncertain]): A stack of frames to be corrected and
            their uncertaintities.
        dark_current (float): The dark current flux.

    Returns:
        ndarray[FrameShape, Uncertain]: The corrected stack of frames and their updated
            uncertainties.
    """
    return subtract_uncertain(frames, array([(dark_current, 0.0)], dtype=uncertain))
