from typing import Any, Tuple, TypeVar

from numpy import cos, empty, exp
from numpy.ma import MaskedArray, masked_array

from degradation_eda.processing_stages.common import scattering_angles
from degradation_eda.utils.uncertain_maths import Uncertain, divide_uncertain, uncertain

FramesShape = TypeVar("FramesShape", bound=Any)


def correct_angular_efficincy(
    frames: MaskedArray[FramesShape, Uncertain],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    absorbtion_coefficient: float,
    thickness: float,
) -> MaskedArray[FramesShape, Uncertain]:
    """Corrects for loss due to the angular efficiency of the detector head.

    Args:
        frames (MaskedArray[FramesShape, Uncertain]): A stack of uncertain frames to be
            corrected.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample head.
        absorbtion_coefficient (float): The coefficient of absorbtion for a given
            material at a given photon energy.
        thickness (float): The thickness of the detector head material.

    Returns:
        MaskedArray[FramesShape, Uncertain]: The corrected stack of frames and their
            updated uncertainties.
    """
    absorbtion_efficiency = empty(frames.shape[1:], uncertain)
    absorbtion_efficiency["nominal"] = 1 - exp(
        -absorbtion_coefficient
        * thickness
        / cos(scattering_angles(frames[0].shape, beam_center, pixel_sizes, distance))
    )
    absorbtion_efficiency["uncertainty"] = 0
    return masked_array(divide_uncertain(frames, absorbtion_efficiency), frames.mask)
