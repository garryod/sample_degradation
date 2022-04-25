from typing import Tuple, TypeVar

from numpy import dtype, empty, floating, ndarray
from numpy.ma import MaskedArray, masked_array

from degradation_eda.utils.uncertain_maths import (
    Uncertain,
    multiply_uncertain,
    uncertain,
)

NumFrames = TypeVar("NumFrames", bound=int)
FrameWidth = TypeVar("FrameWidth", bound=int)
FrameHeight = TypeVar("FrameHeight", bound=int)


def correct_flatfield(
    frames: MaskedArray[Tuple[NumFrames, FrameHeight, FrameWidth], Uncertain],
    flatfield: ndarray[Tuple[FrameHeight, FrameWidth], dtype[floating]],
) -> MaskedArray[Tuple[NumFrames, FrameHeight, FrameWidth], Uncertain]:
    """Apply multiplicative flatfield correction, to correct for inter-pixel sensitivity.

    Apply multiplicative flatfield correction, to correct for inter-pixel sensitivity,
    as described in section 3.xii of 'The modular small-angle X-ray scattering data
    correction sequence' [https://doi.org/10.1107/S1600576717015096].

    Args:
        frames (MaskedArray[Tuple[NumFrames, FrameHeight, FrameWidth], Uncertain]): A
            stack of uncertain frames to be corrected.
        flatfield (ndarray[Tuple[FrameHeight, FrameWidth], dtype[floating]]): The
            multiplicative flatfield correction to be applied.

    Returns:
        MaskedArray[Tuple[NumFrames, FrameHeight, FrameWidth], Uncertain]: _description_
    """
    uncertain_flatfield = empty(flatfield.shape, dtype=uncertain)
    uncertain_flatfield["nominal"] = flatfield
    uncertain_flatfield["uncertainty"] = 0
    return masked_array(
        multiply_uncertain(frames.data, uncertain_flatfield), frames.mask
    )
