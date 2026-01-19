"""
Test normalization behavior for temporal sequences in UrbanHeatDataset.

Ensures that the _normalize method correctly handles both 3D (C, H, W) and 4D (T, C, H, W) data.
"""

import numpy as np

from urbanai.training.dataset import UrbanHeatDataset


class TestDatasetNormalization:
    """Test suite for dataset normalization with temporal sequences."""

    def test_temporal_normalization_zscore(self):
        """Test zscore normalization handles temporal sequences correctly."""
        # Create mock temporal data (T, C, H, W)
        temporal_data = np.random.randn(4, 5, 100, 100).astype(np.float32)

        # Create mock stats with shape (5, 1) as created by _calculate_stats
        stats = {
            "mean": np.random.randn(5, 1).astype(np.float32),
            "std": np.ones((5, 1), dtype=np.float32)
        }

        # Create a mock dataset instance (we only need the _normalize method)
        # We can't fully instantiate without real data files
        class MockDataset:
            def __init__(self):
                self.stats = stats
                self.normalization_method = "zscore"

            # Copy the _normalize method from UrbanHeatDataset
            _normalize = UrbanHeatDataset._normalize

        dataset = MockDataset()
        normalized = dataset._normalize(temporal_data)

        # Verify shape is preserved
        assert normalized.shape == temporal_data.shape, \
            f"Expected shape {temporal_data.shape}, got {normalized.shape}"

        # Verify no NaNs or Infs
        assert not np.isnan(normalized).any(), "Normalized data contains NaNs"
        assert not np.isinf(normalized).any(), "Normalized data contains Infs"

    def test_temporal_normalization_minmax(self):
        """Test minmax normalization handles temporal sequences correctly."""
        # Create mock temporal data (T, C, H, W)
        temporal_data = np.random.randn(3, 7, 50, 50).astype(np.float32)

        # Create mock stats
        stats = {
            "min": np.random.randn(7, 1).astype(np.float32),
            "range": np.ones((7, 1), dtype=np.float32)
        }

        class MockDataset:
            def __init__(self):
                self.stats = stats
                self.normalization_method = "minmax"

            _normalize = UrbanHeatDataset._normalize

        dataset = MockDataset()
        normalized = dataset._normalize(temporal_data)

        assert normalized.shape == temporal_data.shape
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()

    def test_temporal_normalization_robust(self):
        """Test robust normalization handles temporal sequences correctly."""
        # Create mock temporal data (T, C, H, W)
        temporal_data = np.random.randn(5, 5, 75, 75).astype(np.float32)

        # Create mock stats
        stats = {
            "median": np.random.randn(5, 1).astype(np.float32),
            "iqr": np.ones((5, 1), dtype=np.float32)
        }

        class MockDataset:
            def __init__(self):
                self.stats = stats
                self.normalization_method = "robust"

            _normalize = UrbanHeatDataset._normalize

        dataset = MockDataset()
        normalized = dataset._normalize(temporal_data)

        assert normalized.shape == temporal_data.shape
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()

    def test_single_frame_normalization_zscore(self):
        """Test zscore normalization still works for single frames (C, H, W)."""
        # Create mock single frame data (C, H, W)
        single_frame = np.random.randn(5, 100, 100).astype(np.float32)

        # Create mock stats
        stats = {
            "mean": np.random.randn(5, 1).astype(np.float32),
            "std": np.ones((5, 1), dtype=np.float32)
        }

        class MockDataset:
            def __init__(self):
                self.stats = stats
                self.normalization_method = "zscore"

            _normalize = UrbanHeatDataset._normalize

        dataset = MockDataset()
        normalized = dataset._normalize(single_frame)

        # Verify shape is preserved
        assert normalized.shape == single_frame.shape
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()

    def test_broadcasting_correctness_temporal(self):
        """Test that broadcasting works correctly for temporal sequences."""
        # Create specific test case from the bug report
        temporal_data = np.ones((4, 5, 100, 100), dtype=np.float32) * 10.0

        # Mean of 5, std of 2
        stats = {
            "mean": np.ones((5, 1), dtype=np.float32) * 5.0,
            "std": np.ones((5, 1), dtype=np.float32) * 2.0
        }

        class MockDataset:
            def __init__(self):
                self.stats = stats
                self.normalization_method = "zscore"

            _normalize = UrbanHeatDataset._normalize

        dataset = MockDataset()
        normalized = dataset._normalize(temporal_data)

        # (10 - 5) / 2 = 2.5 for all values
        expected = 2.5
        assert normalized.shape == (4, 5, 100, 100)
        assert np.allclose(normalized, expected), \
            f"Expected all values to be {expected}, got range [{normalized.min()}, {normalized.max()}]"

    def test_no_stats_returns_unchanged(self):
        """Test that data is returned unchanged when stats is None."""
        temporal_data = np.random.randn(4, 5, 100, 100).astype(np.float32)

        class MockDataset:
            def __init__(self):
                self.stats = None
                self.normalization_method = "zscore"

            _normalize = UrbanHeatDataset._normalize

        dataset = MockDataset()
        result = dataset._normalize(temporal_data)

        assert np.array_equal(result, temporal_data), "Data should be unchanged when stats is None"


if __name__ == "__main__":
    # Run tests manually
    test = TestDatasetNormalization()

    print("Running test_temporal_normalization_zscore...")
    test.test_temporal_normalization_zscore()
    print("✓ Passed")

    print("Running test_temporal_normalization_minmax...")
    test.test_temporal_normalization_minmax()
    print("✓ Passed")

    print("Running test_temporal_normalization_robust...")
    test.test_temporal_normalization_robust()
    print("✓ Passed")

    print("Running test_single_frame_normalization_zscore...")
    test.test_single_frame_normalization_zscore()
    print("✓ Passed")

    print("Running test_broadcasting_correctness_temporal...")
    test.test_broadcasting_correctness_temporal()
    print("✓ Passed")

    print("Running test_no_stats_returns_unchanged...")
    test.test_no_stats_returns_unchanged()
    print("✓ Passed")

    print("\nAll tests passed! ✓")
