from typing import Tuple, TypeVar

from numpy import cos, empty, sin, square
from numpy.ma import MaskedArray, masked_array

from degradation_eda.processing_stages.common import azimuthal_angles, scattering_angles
from degradation_eda.utils.uncertain_maths import (
    Uncertain,
    multiply_uncertain,
    uncertain,
)

FramesShape = TypeVar("FramesShape", bound=Tuple[int, int, int])


def correct_polarization(
    frames: MaskedArray[FramesShape, Uncertain],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    horizontal_poarization: float = 0.5,
) -> MaskedArray[FramesShape, Uncertain]:
    """Corrects for the effect of polarization of the incident beam.

    Corrects for the effect of polarization of the incident beam, as detailed in
    section 3.4.1 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (MaskedArray[FramesShape, Uncertain]): A stack of uncertain frames to be
            corrected.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample.
        horizontal_poarization (float, optional): The fraction of incident radiation
            polarized in the horizontal plane, where 0.5 signifies an unpolarized
            source. Defaults to 0.5.

    Returns:
        MaskedArray[FramesShape, Uncertain]: _description_
    """
    scattering = scattering_angles(
        frames.shape[-2:], beam_center, pixel_sizes, distance
    )
    azimuths = azimuthal_angles(frames.shape[-2:], beam_center)
    correction_factor = empty(frames.shape[1:], dtype=uncertain)
    correction_factor["nominal"] = horizontal_poarization * (
        1 - square(sin(azimuths) * sin(scattering))
    ) + (1 - horizontal_poarization) * (1 - square(cos(azimuths) * sin(scattering)))
    correction_factor["uncertainty"] = 0
    return masked_array(
        multiply_uncertain(frames, correction_factor),
        frames.mask,
    )
