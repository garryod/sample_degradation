from numpy import floating, maximum, number, sqrt
from numpy.typing import NDArray


def estimate_poisson_uncertainty(frames: NDArray[number]) -> NDArray[floating]:
    """Estimates the poisson uncertainty of photon counts.

    Estimates the poisson uncertainty of photon counts by taking the square root of
    real numbers greater than one, or one in place of numbers less than one.

    Args:
        frames (NDArray[number]): A stack of frames from which the uncertainties can be
            computed.

    Returns:
        NDArray[floating]: The uncertainties associated with pixels in the frame stack.
    """
    return maximum(sqrt(frames), 1.0)
