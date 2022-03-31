from numpy import floating
from numpy.typing import NDArray
from plotly.express import imshow


def show_frame(frame: NDArray[floating]):
    """Generates and displays a plotly heatmap from a 2 dimensional frame.

    Args:
        frame (NDArray[floating]): A 2 dimensional frame.
    """
    imshow(frame).show()
