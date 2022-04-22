from typing import Tuple, TypeVar

from numpy import arccos, empty
from numpy.ma import MaskedArray, masked_array

from degradation_eda.processing_stages.common import scattering_angles
from degradation_eda.utils.uncertain_maths import Uncertain, divide_uncertain, uncertain

FramesShape = TypeVar("FramesShape", bound=Tuple[int, int, int])


def correct_solid_angle(
    frames: MaskedArray[FramesShape, Uncertain],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
) -> MaskedArray[FramesShape, Uncertain]:
    """Corrects for the solid angle subtended by each pixel.

    Args:
        frames (MaskedArray[FramesShape, Uncertain]): A stack of uncertain frames to be
            corrected.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample head.

    Returns:
        MaskedArray[FramesShape, Uncertain]: The corrected stack of frames and their
            updated uncertainties.
    """
    correction = empty(frames.shape[1:], dtype=uncertain)
    correction["nominal"] = arccos(
        scattering_angles(frames.shape[1:], beam_center, pixel_sizes, distance)
    )
    correction["uncertainty"] = 0
    return masked_array(
        divide_uncertain(frames, correction),
        frames.mask,
    )
