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

            return core_schema.no_info_plain_validator_function(validate)

    return _NDArray


FloatMatrix = NDArray(np.float64, (2, 2))


class Model(BaseModel):
    x: FloatMatrix


m = Model(x=[[1.0, 2.0], [3.0, 4.0]])
print(m.x)
print(type(m.x))
