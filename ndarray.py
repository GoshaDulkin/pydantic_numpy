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
    dtype and optional shape validation. Doesn't support
    symbolic or wildcard dimensions.
    """
    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}.")

    class PydanticNDArray:
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type, handler: GetCoreSchemaHandler
        ):
            def validate(value: Any) -> np.ndarray:
                arr = np.array(value)
                actual_dtype = arr.dtype
                actual_shape = arr.shape

                if actual_dtype != np.dtype(dtype):
                    raise ValueError(f"Expected dtype {dtype}, got {actual_dtype}")

                if shape is not None and actual_shape != shape:
                    raise ValueError(f"Expected shape {shape}, got {actual_shape}")

                return arr

            def serialize(value: np.ndarray) -> Any:
                return value.tolist()

            return core_schema.no_info_plain_validator_function(
                validate,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    serialize
                ),
            )

    return PydanticNDArray
