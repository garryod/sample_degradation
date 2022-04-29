from typing import Tuple, TypeVar

from numpy import empty
from numpy.ma import MaskedArray, masked_array

from sample_degradation.utils.uncertain_maths import (
    Uncertain,
    multiply_uncertain,
    uncertain,
)

FramesShape = TypeVar("FramesShape", bound=Tuple[int, int, int])


def correct_displaced_volume(
    frames: MaskedArray[FramesShape, Uncertain],
    displaced_fraction: float,
) -> MaskedArray[FramesShape, Uncertain]:
    """Correct for displaced volume of solvent by multiplying signal by retained fraction.

    Correct for displaced volume of solvent by multiplying signal by the retained
    fraction, as detailed in section 3.xviii and appendix B of `The modular small-angle
    X-ray scattering data correction sequence'
    [https://doi.org/10.1107/S1600576717015096].

    Args:
        frames (MaskedArray[FramesShape, Uncertain]):  A stack of uncertain frames to be
            corrected.
        displaced_fraction (float): The fraction of solvent displaced by the analyte.

    Returns:
        MaskedArray[FramesShape, Uncertain]: The corrected stack of frames and their
            updated uncertainties.
    """
    retained_fraction = empty((1,), dtype=uncertain)
    retained_fraction["nominal"] = 1 - displaced_fraction
    retained_fraction["uncertainty"] = 0
    return masked_array(multiply_uncertain(frames.data, retained_fraction))
