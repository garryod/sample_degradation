from numpy import number
from numpy.typing import NDArray
from plotly.express import imshow
from plotly.express.colors import get_colorscale


def show_frame(frame: NDArray[number], log_z: float = 1.0):
    """Generates and displays a plotly heatmap from a 2 dimensional frame.

    Args:
        frame (NDArray[number]): A 2 dimensional frame.
        log_z (float): The exponential scaling to apply on the z axis. Default 1.0.
    """
    BASE_SCALE = get_colorscale("viridis")
    color_scale = (
        [
            [(log_z**val - log_z**0) / (log_z**1 - log_z**0), color]
            for val, color in BASE_SCALE
        ]
        if log_z > 1
        else BASE_SCALE
    )
    imshow(frame, color_continuous_scale=color_scale).show()
