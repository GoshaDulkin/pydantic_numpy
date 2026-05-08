import numpy as np
from typing import Any
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

ALLOWED_DTYPES = {
    np.int64,
    np.float64,
    np.complex128,
}


def NDArray(dtype: type[np.generic], shape: tuple[int, ...] | None = None) -> type:
    """
    Create a Pydantic-compatible NumPy ndarray type with
    dtype and optional shape validation.

    Supports exact shape matching only; symbolic and wildcard
    dimensions are intentionally not implemented.
    """
    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}.")

    expected_dtype = np.dtype(dtype)

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
                "Complex array must be provided as " "{'real': ..., 'imag': ...}"
            ) from e

        if real.shape != imag.shape:
            raise ValueError("Real and imaginary parts must have matching shapes.")

        return real + 1j * imag

    class PydanticNDArray:
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type,
            handler: GetCoreSchemaHandler,
        ):
            def validate(value: Any) -> np.ndarray:
                if np.issubdtype(dtype, np.complexfloating) and isinstance(value, dict):
                    value = deserialize_complex(value)

                arr = np.asarray(value)

                if arr.dtype != expected_dtype:
                    raise ValueError(
                        f"Expected dtype {expected_dtype}, got {arr.dtype}"
                    )

                if shape is not None and arr.shape != shape:
                    raise ValueError(f"Expected shape {shape}, got {arr.shape}")

                return arr

            def serialize(value: np.ndarray) -> Any:
                if np.issubdtype(value.dtype, np.complexfloating):
                    return {
                        "real": np.real(value).tolist(),
                        "imag": np.imag(value).tolist(),
                    }

                return value.tolist()

            return core_schema.no_info_plain_validator_function(
                validate,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    serialize
                ),
            )

    return PydanticNDArray
