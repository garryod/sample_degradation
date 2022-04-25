from typing import Any, Tuple, TypeVar, cast

from numpy import cos, dtype, empty, floating, log, ndarray, power
from numpy.ma import MaskedArray, masked_array

from degradation_eda.processing_stages.common import scattering_angles
from degradation_eda.utils.uncertain_maths import (
    Uncertain,
    multiply_uncertain,
    uncertain,
)

FramesShape = TypeVar("FramesShape", bound=Any)


def self_absorbtion_correction_factors(
    frame_shape: Tuple[int, int],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    transmissibility: float,
) -> ndarray[Any, dtype[floating]]:
    """Computes the self absorbtion correction factors given geometry and transmissibility.

    Args:
        frame_shape (Tuple[int, int]): The shape of a frame.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample.
        transmissibility (float): The transmissibility of the sample.

    Returns:
        ndarray[Any, dtype[floating]]: An array of correction factors to be applied to
            frames.
    """
    angles = scattering_angles(frame_shape, beam_center, pixel_sizes, distance)
    return (1 - power(transmissibility, 1 / cos(angles) - 1)) / (
        log(transmissibility) * (1 - 1 / cos(angles))
    )


def correct_self_absorbtion(
    frames: MaskedArray[FramesShape, Uncertain],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    transmissibility: float,
) -> MaskedArray[FramesShape, Uncertain]:
    """Correct for transmission loss due to differences in observation angle.

    Correct for transmission loss due to differences in observation angle, as detailed
    in section 3.4.7 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (MaskedArray[FramesShape, Uncertain]): A stack of uncertain frames to be
            corrected.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample.
        transmissibility (float): The transmissibility of the sample.

    Returns:
        MaskedArray[FramesShape, Uncertain]: The corrected stack of frames and their
            updated uncertainties.
    """
    correction_factors = empty(frames.shape[1:], uncertain)
    correction_factors["nominal"] = self_absorbtion_correction_factors(
        cast(Tuple[int, int], frames.shape[1:]),
        beam_center,
        pixel_sizes,
        distance,
        transmissibility,
    )
    correction_factors["uncertainty"] = 0
    return masked_array(
        multiply_uncertain(frames.data, correction_factors), frames.mask
    )
