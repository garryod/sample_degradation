from typing import Any, TypeVar

from numpy import ndarray

from degradation_eda.utils.uncertain_maths import (
    Uncertain,
    divide_uncertain,
    sum_uncertain,
)

FramesShape = TypeVar("FramesShape", bound=Any)


def correct_flux_and_transmission(
    frames: ndarray[FramesShape, Uncertain]
) -> ndarray[FramesShape, Uncertain]:
    """Correct for incident flux and transmissibility by scaling photon counts.

    Correct for incident flux and transmissibility by scaling photon counts with
    respect to the total observed flux.

    Args:
        frames (ndarray[FramesShape, Uncertain]): A stack of uncertain frames to be
            corrected.

    Returns:
        ndarray[FramesShape, Uncertain]: The corrected stack of frames and their
            updated uncertainties.
    """
    frame_flux = sum_uncertain(frames, axis=(1, 2))[:, None, None]
    return divide_uncertain(frames, frame_flux)
