import numpy as np
import pytest

from pydantic import BaseModel

from ndarray import NDArray

FloatMatrix = NDArray(np.float64, (2, 2))


class TestNDArray:
    """Test suite for custom Pydantic-compatible NumPy ndarray types."""

    # ------------------------------------------------------------------
    # Type construction
    # ------------------------------------------------------------------

    def test_unsupported_dtype(self):
        """Raise an error for unsupported NumPy dtypes."""

        with pytest.raises(ValueError, match="Unsupported dtype"):
            NDArray(np.float32)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_valid_array(self):
        """Validate successful creation of a correctly typed and shaped array."""

        class ModelLocal(BaseModel):
            x: FloatMatrix

        m = ModelLocal(x=[[1.0, 2.0], [3.0, 4.0]])

        assert isinstance(m.x, np.ndarray)
        assert m.x.shape == (2, 2)
        assert m.x.dtype == np.float64

    def test_invalid_shape(self):
        """Reject incorrect array shapes."""

        class ModelLocal(BaseModel):
            x: FloatMatrix

        with pytest.raises(ValueError, match="Shape mismatch"):
            ModelLocal(x=[[1.0, 2.0]])

    def test_invalid_rank(self):
        """Reject arrays with incorrect number of dimensions."""

        FloatMatrixWildcard = NDArray(np.float64, (None, 3))

        class ModelLocal(BaseModel):
            x: FloatMatrixWildcard

        # 1D instead of 2D
        with pytest.raises(ValueError, match="Rank mismatch"):
            ModelLocal(x=[1.0, 2.0, 3.0])

    def test_invalid_dtype(self):
        """Raise an error when the provided array dtype is incorrect."""

        class ModelLocal(BaseModel):
            x: FloatMatrix

        with pytest.raises(ValueError, match="Expected dtype"):
            ModelLocal(x=[[1, 2], [3, 4]])

    # ------------------------------------------------------------------
    # Dimensionality
    # ------------------------------------------------------------------

    def test_1d_array(self):
        """Validate support for one-dimensional arrays."""

        FloatVector = NDArray(np.float64, (3,))

        class ModelLocal(BaseModel):
            x: FloatVector

        m = ModelLocal(x=[1.0, 2.0, 3.0])

        assert isinstance(m.x, np.ndarray)
        assert m.x.shape == (3,)
        assert m.x.dtype == np.float64

    def test_scalar_array(self):
        """Validate support for zero-dimensional scalar ndarrays."""

        FloatScalar = NDArray(np.float64, ())

        class ModelLocal(BaseModel):
            x: FloatScalar

        m = ModelLocal(x=1.5)

        assert isinstance(m.x, np.ndarray)
        assert m.x.shape == ()
        assert m.x.dtype == np.float64

        assert m.x.item() == 1.5

    def test_array_without_shape_constraint(self):
        """Allow arrays of arbitrary shape when no shape is specified."""

        FloatArray = NDArray(np.float64)

        class ModelLocal(BaseModel):
            x: FloatArray

        m = ModelLocal(x=[[1.0], [2.0], [3.0]])

        assert isinstance(m.x, np.ndarray)
        assert m.x.shape == (3, 1)
        assert m.x.dtype == np.float64

    def test_wildcard_shape_valid(self):
        """Allow arbitrary size in wildcard dimensions (None)."""

        FloatMatrixWildcard = NDArray(np.float64, (None, 3))

        class ModelLocal(BaseModel):
            x: FloatMatrixWildcard

        m = ModelLocal(x=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        assert isinstance(m.x, np.ndarray)
        assert m.x.shape == (2, 3)
        assert m.x.dtype == np.float64

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_model_dump(self):
        """Serialize ndarray fields into JSON-compatible nested lists."""

        class ModelLocal(BaseModel):
            x: FloatMatrix

        m = ModelLocal(x=[[1.0, 2.0], [3.0, 4.0]])

        dumped = m.model_dump()

        assert dumped == {
            "x": [[1.0, 2.0], [3.0, 4.0]],
        }

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

        np.testing.assert_array_equal(original.x, reconstructed.x)

    # ------------------------------------------------------------------
    # Complex serialization
    # ------------------------------------------------------------------

    def test_complex_model_dump(self):
        """Serialize complex ndarrays into JSON-compatible real/imag parts."""

        ComplexVector = NDArray(np.complex128, (2,))

        class ModelLocal(BaseModel):
            x: ComplexVector

        m = ModelLocal(x=[1 + 2j, 3 + 4j])

        dumped = m.model_dump()

        assert dumped == {
            "x": {
                "real": [1.0, 3.0],
                "imag": [2.0, 4.0],
            }
        }

    def test_complex_json_round_trip(self):
        """Ensure complex ndarrays survive JSON serialization and reconstruction."""

        ComplexVector = NDArray(np.complex128, (2,))

        class ModelLocal(BaseModel):
            x: ComplexVector

        original = ModelLocal(x=[1 + 2j, 3 + 4j])

        json_data = original.model_dump_json()

        reconstructed = ModelLocal.model_validate_json(json_data)

        assert isinstance(reconstructed.x, np.ndarray)
        assert reconstructed.x.dtype == np.complex128

        np.testing.assert_array_equal(original.x, reconstructed.x)
