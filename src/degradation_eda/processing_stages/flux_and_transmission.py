from typing import Any, TypeVar

from numpy import expand_dims
from numpy.ma import MaskedArray, masked_array

from degradation_eda.utils.uncertain_maths import (
    Uncertain,
    divide_uncertain,
    sum_uncertain,
)

FramesShape = TypeVar("FramesShape", bound=Any)


def normalize_transmitted_flux(
    frames: MaskedArray[FramesShape, Uncertain]
) -> MaskedArray[FramesShape, Uncertain]:
    """Normalize for incident flux and transmissibility by scaling photon counts.

    Normalize for incident flux and transmissibility by scaling photon counts with
    respect to the total observed flux.

    Args:
        frames (MaskedArray[FramesShape, Uncertain]): A stack of uncertain frames to be
            normalized.

    Returns:
        MaskedArray[FramesShape, Uncertain]: The normalized stack of frames and their
            updated uncertainties.
    """
    frame_flux = expand_dims(sum_uncertain(frames.filled(0), axis=(1, 2)), (1, 2))
    return masked_array(divide_uncertain(frames.data, frame_flux), frames.mask)
