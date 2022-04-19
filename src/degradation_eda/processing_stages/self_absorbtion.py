from typing import Any, Tuple, TypeVar, cast

from numpy import (
    cos,
    dtype,
    empty,
    floating,
    hypot,
    integer,
    linspace,
    log,
    meshgrid,
    ndarray,
    power,
    tan,
)
from numpy.ma import MaskedArray, masked_array

from degradation_eda.utils.uncertain_maths import (
    Uncertain,
    multiply_uncertain,
    uncertain,
)

FramesShape = TypeVar("FramesShape", bound=Any)


def pixel_angles(
    frame_shape: Tuple[int, int],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
) -> ndarray[Tuple[int, int], dtype[integer]]:
    """Computes the angles of pixels from the sample for a given geometry.

    Args:
        frame_shape (Tuple[int, int]): The shape of a frame.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample.

    Returns:
        ndarray[Tuple[int, int], dtype[integer]]: An array of pixel angles from the
            sample.
    """
    yy, xx = meshgrid(
        linspace(
            -beam_center[1] * pixel_sizes[1],
            (frame_shape[1] - beam_center[1]) * pixel_sizes[1],
            frame_shape[1],
        ),
        linspace(
            -beam_center[0] * pixel_sizes[0],
            (frame_shape[0] - beam_center[0]) * pixel_sizes[0],
            frame_shape[0],
        ),
    )
    return tan(hypot(xx, yy) / distance)


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
    angles = pixel_angles(frame_shape, beam_center, pixel_sizes, distance)
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
