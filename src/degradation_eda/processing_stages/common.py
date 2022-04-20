from typing import Tuple

from numpy import dtype, hypot, integer, linspace, meshgrid, ndarray, tan


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
