import numpy as np
from typing import Any
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema

ALLOWED_DTYPES = {
    np.int64,
    np.float64,
    np.complex128,
}


def NDArray(dtype: type[np.generic], shape: tuple[int, ...] | None = None):
    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}.")

    class _NDArray:
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type, handler: GetCoreSchemaHandler
        ):
            def validate(value: Any) -> np.ndarray:
                arr = np.array(value)

                if arr.dtype != np.dtype(dtype):
                    raise TypeError(f"Expected dtype {dtype}, got {arr.dtype}")

                if shape is not None and arr.shape != shape:
                    raise TypeError(f"Expected shape {shape}, got {arr.shape}")

                return arr

            def serialize(value: np.ndarray) -> Any:
                return value.tolist()

            return core_schema.no_info_plain_validator_function(
                validate,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    serialize
                ),
            )

    return _NDArray
