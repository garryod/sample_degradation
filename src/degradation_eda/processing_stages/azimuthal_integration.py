from typing import Any, Tuple, TypeVar, cast

from numpy import divide, dtype, floating, histogram, int_, ndarray
from numpy.ma import MaskedArray, masked_where

from degradation_eda.processing_stages.self_absorbtion import scattering_angles

FrameWidth = TypeVar("FrameWidth", bound=int)
FrameHeight = TypeVar("FrameHeight", bound=int)
AzimuthShape = TypeVar(
    "AzimuthShape",
    bound=Tuple[
        int,
    ],
)


def azimuthally_integrate(
    frame: MaskedArray[Tuple[FrameWidth, FrameHeight], dtype[floating]],
    beam_center: Tuple[float, float],
    pixel_sizes: Tuple[float, float],
    distance: float,
    num_bins: int = 1000,
) -> Tuple[ndarray[AzimuthShape, dtype[Any]], ndarray[AzimuthShape, dtype[floating]]]:
    """Performs azimuthal integration of the masked frame, producing an intensity vector.

    Performs azimuthal integration of the masked frame via angular binning, producing an
    intensity vector with respect to azimuth and the corresponding azimutal centroid
    vector.

    Args:
        frame (MaskedArray[Tuple[FrameWidth, FrameHeight], dtype[floating]]): A frame
            containing pixel intensities.
        beam_center (Tuple[float, float]): The center position of the beam in pixels.
        pixel_sizes (Tuple[float, float]): The real space size of a detector pixel.
        distance (float): The distance between the detector and the sample.
        num_bins (int, optional): The number of bins to separate azimuthal intensiities
            into. Defaults to 1000.

    Returns:
        Tuple[
            ndarray[AzimuthShape, dtype[Any]],
            ndarray[AzimuthShape, dtype[floating]]
        ]: A tuple containing the azimuthal intensities at a set of linearly spaced
            azimuths and those corresponding azimuthal centroids.
    """
    angles: MaskedArray[Tuple[FrameWidth, FrameHeight], dtype[floating]] = masked_where(
        frame.mask,
        scattering_angles(
            cast(Tuple[FrameWidth, FrameHeight], frame.shape),
            beam_center,
            pixel_sizes,
            distance,
        ),
    )
    range = (
        angles.min(),
        angles.max(),
    )
    hist, edges = histogram(
        angles.filled(0.0), weights=frame.filled(0.0), bins=num_bins, range=range
    )
    norm, _ = histogram(
        angles.filled(0.0),
        weights=(1 - frame.mask.astype(int_)),
        bins=num_bins,
        range=range,
    )
    azimuthal_intensity = divide(hist, norm)
    azimuth = (edges[1:] + edges[:-1]) / 2
    return azimuthal_intensity, azimuth
