import pytest
import numpy as np

from pydantic import BaseModel
from ndarray import NDArray

FloatMatrix = NDArray(np.float64, (2, 2))


class TestNDArray:
    def test_valid_array(self):
        class ModelLocal(BaseModel):
            x: FloatMatrix

        m = ModelLocal(x=[[1.0, 2.0], [3.0, 4.0]])

        assert isinstance(m.x, np.ndarray)
        assert m.x.shape == (2, 2)
        assert m.x.dtype == np.float64

    def test_invalid_shape(self):
        class ModelLocal(BaseModel):
            x: FloatMatrix

        with pytest.raises(TypeError):
            ModelLocal(x=[[1.0, 2.0]])  # wrong shape

    def test_invalid_dtype(self):
        class ModelLocal(BaseModel):
            x: FloatMatrix

        with pytest.raises(TypeError):
            ModelLocal(x=[[1, 2], [3, 4]])  # int instead of float64
