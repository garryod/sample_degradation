from typing import Any, Sequence, SupportsIndex, TypeVar, Union

from numpy import (
    add,
    broadcast_shapes,
    divide,
    dtype,
    empty,
    float_,
    hypot,
    multiply,
    ndarray,
    sqrt,
    square,
    stack,
    subtract,
    sum,
    void,
)
from numpy.lib.recfunctions import unstructured_to_structured

ShapeA = TypeVar("ShapeA", bound=Any)
ShapeB = TypeVar("ShapeB", bound=Any)

uncertain = dtype([("nominal", float_), ("uncertainty", float_)])
Uncertain = dtype[void]


def sum_uncertain(
    arr: ndarray[Any, Uncertain],
    axis: Union[None, SupportsIndex, Sequence[SupportsIndex]] = None,
) -> ndarray[Any, Uncertain]:
    """Sum the nominal components of an uncertain array and compute uncertainty.

    Sum the nominal components of an uncertain array and compute uncertainty as the
    magnitude of a vector between the sum of orthognal vectors with magnitudes
    corresponding to array elements.

    Args:
        arr (ndarray[Any, Uncertain]): The uncertain array to be summed.
        axis (Union[None, SupportsIndex, Sequence[SupportsIndex]]): The axis or axis
            along which the sum is performed, where None denotes a sum across all axis.
            Default None.

    Returns:
        ndarray[Any, Uncertain]: The sum with an associated uncertainty.
    """
    return unstructured_to_structured(
        stack(
            [
                sum(arr["nominal"], axis),  # type: ignore
                sqrt(sum(square(arr["uncertainty"]), axis)),  # type: ignore
            ],
            axis=-1,
        ),
        uncertain,
    )


def add_uncertain(
    augend: ndarray[ShapeA, Uncertain], addend: ndarray[ShapeB, Uncertain]
) -> ndarray[Union[ShapeA, ShapeB], Uncertain]:
    """Add uncertain arrays element wise and compute uncertainty.

    Args:
        augend (ndarray[ShapeA, Uncertain]): The first uncertain array to be added.
        addend (ndarray[ShapeB, Uncertain]): The second uncertain array to be added.

    Returns:
        ndarray[Union[ShapeA, ShapeB], Uncertain]: The element wise sum of arrays with
            an associated element wise uncertainity.
    """
    result = empty(broadcast_shapes(augend.shape, addend.shape), uncertain)
    result["nominal"] = add(augend["nominal"], addend["nominal"])
    result["uncertainty"] = hypot(augend["uncertainty"], addend["uncertainty"])
    return result


def subtract_uncertain(
    minuend: ndarray[ShapeA, Uncertain], subtrahend: ndarray[ShapeB, Uncertain]
) -> ndarray[Union[ShapeA, ShapeB], Uncertain]:
    """Subtract uncertain arrays element wise and compute uncertainty.

    Args:
        minuend (ndarray[ShapeA, Uncertain]): The uncertain array to be subtracted from.
        subtrahend (ndarray[ShapeB, Uncertain]): The uncertain array to subtract.

    Returns:
        ndarray[Union[ShapeA, ShapeB], Uncertain]: The element wise subtraction of
            arrays with an associated element wise uncertainity.
    """
    result = empty(broadcast_shapes(minuend.shape, subtrahend.shape), uncertain)
    result["nominal"] = subtract(minuend["nominal"], subtrahend["nominal"])
    result["uncertainty"] = hypot(minuend["uncertainty"], subtrahend["uncertainty"])
    return result


def multiply_uncertain(
    multiplier: ndarray[ShapeA, Uncertain], multiplicand: ndarray[ShapeB, Uncertain]
) -> ndarray[Union[ShapeA, ShapeB], Uncertain]:
    """Mutliply uncertain arrays element wise and compute uncertainity.

    Args:
        arr_a (ndarray[ShapeA, Uncertain]): The first uncertain array to be multiplied.
        arr_b (ndarray[ShapeB, Uncertain]): The second uncertain array to be multiplied.

    Returns:
        ndarray[Union[ShapeA, ShapeB], Uncertain]: The element wise mutltiplication of
            arrays with an associated element wise uncertainity.
    """
    result = empty(broadcast_shapes(multiplier.shape, multiplicand.shape), uncertain)
    result["nominal"] = multiply(multiplier["nominal"], multiplicand["nominal"])
    result["uncertainty"] = hypot(
        multiply(multiplier["uncertainty"], multiplicand["nominal"]),
        multiply(multiplicand["uncertainty"], multiplier["nominal"]),
    )
    return result


def divide_uncertain(
    dividend: ndarray[ShapeA, Uncertain], divisor: ndarray[ShapeB, Uncertain]
) -> ndarray[Union[ShapeA, ShapeB], Uncertain]:
    """Divide uncertain arrays element wise and compute uncertainity.

    Args:
        dividend (ndarray[ShapeA, Uncertain]): The uncertain array of enumerators.
        divisor (ndarray[ShapeB, Uncertain]): The uncertain array of denomenatiors.

    Returns:
        ndarray[Union[ShapeA, ShapeB], Uncertain]: _description_
    """
    result = empty(broadcast_shapes(dividend.shape, divisor.shape), uncertain)
    result["nominal"] = divide(dividend["nominal"], divisor["nominal"])
    result["uncertainty"] = hypot(
        dividend["uncertainty"] / divisor["nominal"],
        divisor["uncertainty"] * dividend["nominal"] / square(divisor["nominal"]),
    )
    return result
