import pytest
import numpy as np

from pydantic import BaseModel
from ndarray import NDArray

FloatMatrix = NDArray(np.float64, (2, 2))


class TestNDArray:
    """Test suite for custom Pydantic-compatible NumPy ndarray types."""

    def test_unsupported_dtype(self):
        """Raise an error for unsupported NumPy dtypes."""

        with pytest.raises(ValueError, match="Unsupported dtype"):
            NDArray(np.float32)

    def test_valid_array(self):
        """Validate successful creation of a correctly typed and shaped array."""

        class ModelLocal(BaseModel):
            x: FloatMatrix

        m = ModelLocal(x=[[1.0, 2.0], [3.0, 4.0]])

        assert isinstance(m.x, np.ndarray)
        assert m.x.shape == (2, 2)
        assert m.x.dtype == np.float64

    def test_invalid_shape(self):
        """Raise an error when the provided array shape is incorrect."""

        class ModelLocal(BaseModel):
            x: FloatMatrix

        with pytest.raises(TypeError, match="Expected shape"):
            ModelLocal(x=[[1.0, 2.0]])

    def test_invalid_dtype(self):
        """Raise an error when the provided array dtype is incorrect."""

        class ModelLocal(BaseModel):
            x: FloatMatrix

        with pytest.raises(TypeError, match="Expected dtype"):
            ModelLocal(x=[[1, 2], [3, 4]])

    def test_model_dump(self):
        """Serialize ndarray fields into JSON-compatible nested lists."""

        class ModelLocal(BaseModel):
            x: FloatMatrix

        m = ModelLocal(x=[[1.0, 2.0], [3.0, 4.0]])

        dumped = m.model_dump()

        assert dumped == {"x": [[1.0, 2.0], [3.0, 4.0]]}

    def test_json_round_trip(self):
        """Ensure ndarray values survive JSON serialization and reconstruction."""

        class ModelLocal(BaseModel):
            x: FloatMatrix

        original = ModelLocal(x=[[1.0, 2.0], [3.0, 4.0]])

        json_data = original.model_dump_json()

        reconstructed = ModelLocal.model_validate_json(json_data)

        assert isinstance(reconstructed.x, np.ndarray)
        assert reconstructed.x.shape == (2, 2)
        assert reconstructed.x.dtype == np.float64
        assert np.array_equal(original.x, reconstructed.x)

    def test_int64_array(self):
        """Validate support for int64 ndarray types."""

        IntVector = NDArray(np.int64, (3,))

        class ModelLocal(BaseModel):
            x: IntVector

        m = ModelLocal(x=[1, 2, 3])

        assert isinstance(m.x, np.ndarray)
        assert m.x.shape == (3,)
        assert m.x.dtype == np.int64

    def test_1d_array(self):
        """Validate support for one-dimensional arrays."""

        FloatVector = NDArray(np.float64, (3,))

        class ModelLocal(BaseModel):
            x: FloatVector

        m = ModelLocal(x=[1.0, 2.0, 3.0])

        assert m.x.shape == (3,)
        assert m.x.dtype == np.float64

    def test_array_without_shape_constraint(self):
        """Allow arrays of arbitrary shape when no shape is specified."""

        FloatArray = NDArray(np.float64)

        class ModelLocal(BaseModel):
            x: FloatArray

        m = ModelLocal(x=[[1.0], [2.0], [3.0]])

        assert isinstance(m.x, np.ndarray)
        assert m.x.shape == (3, 1)
        assert m.x.dtype == np.float64
