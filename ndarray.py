import numpy as np
from typing import Any
from pydantic import GetCoreSchemaHandler, ValidationInfo
from pydantic_core import core_schema

ALLOWED_DTYPES = {
    np.int64,
    np.float64,
    np.complex128,
}


def NDArray(
    dtype: type[np.number], shape: tuple[int | None, ...] | None = None
) -> type:
    """
    Create a Pydantic-compatible NumPy ndarray type with
    dtype and optional shape validation.

    Supports:
    - exact shape matching
    - wildcard dimensions using 'None'
    """
    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}.")

    expected_dtype = np.dtype(dtype)

    def validate_shape(actual_shape: tuple[int, ...]) -> None:
        """
        Validate an ndarray shape against the expected shape.

        Rules:
        - If shape is None: no validation is performed
        - If shape contains None: that dimension is a wildcard
          (matches any size)
        - Otherwise dimensions must match exactly

        Example:
            shape = (None, 3)
            valid shapes: (10, 3), (1, 3)
            invalid shapes: (3, 1), (10, 2)
        """
        if shape is None:
            return

        if len(actual_shape) != len(shape):
            raise ValueError(
                "Rank mismatch: "
                f"expected {len(shape)} dimensions, "
                f"got {len(actual_shape)} with shape {actual_shape}"
            )

        for i, (actual_dim, expected_dim) in enumerate(zip(actual_shape, shape)):
            if expected_dim is not None and actual_dim != expected_dim:
                raise ValueError(
                    f"Shape mismatch at dim {i}: expected {expected_dim}, got {actual_dim}"
                )

    def deserialize_complex(value: dict[str, Any]) -> np.ndarray:
        """
        Reconstruct a complex ndarray from a JSON-compatible
        {"real": ..., "imag": ...} representation.
        """
        try:
            real = np.asarray(value["real"])
            imag = np.asarray(value["imag"])
        except (KeyError, TypeError) as e:
            raise ValueError(
                "Complex array must be provided as {'real': ..., 'imag': ...}"
            ) from e

        if real.shape != imag.shape:
            raise ValueError("Real and imaginary parts must have matching shapes.")

        return real + 1j * imag

    class PydanticNDArray:
        """Dynamically generated Pydantic-compatible ndarray type."""

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type,
            handler: GetCoreSchemaHandler,
        ):
            def validate(value: Any, info: ValidationInfo) -> np.ndarray:
                if np.issubdtype(dtype, np.complexfloating) and isinstance(value, dict):
                    value = deserialize_complex(value)

                arr = np.asarray(value)

                if arr.dtype != expected_dtype:
                    raise ValueError(
                        f"Expected dtype {expected_dtype}, got {arr.dtype}"
                    )

                validate_shape(arr.shape)

                return arr

            def serialize(value: np.ndarray) -> Any:
                if np.issubdtype(value.dtype, np.complexfloating):
                    return {
                        "real": np.real(value).tolist(),
                        "imag": np.imag(value).tolist(),
                    }

                return value.tolist()

            return core_schema.with_info_plain_validator_function(
                validate,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    serialize
                ),
            )

    return PydanticNDArray
