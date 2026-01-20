"""
Test data augmentation shape consistency for batched training.

Ensures that augmentation operations maintain consistent shapes across
samples in a batch, preventing DataLoader collation errors.
"""

import numpy as np

from urbanai.training.dataset import UrbanHeatDataset


class TestAugmentationShapeConsistency:
    """Test suite for augmentation shape consistency."""

    def test_augmentation_preserves_shape(self):
        """Test that augmentation always preserves input shape."""
        # Create mock temporal sequences (T, C, H, W)
        input_seq = np.random.randn(4, 5, 1047, 439).astype(np.float32)
        target_seq = np.random.randn(4, 5, 1047, 439).astype(np.float32)
        
        # Create augmentation config with all operations enabled
        augment_config = {
            "horizontal_flip_prob": 1.0,  # Force to happen
            "vertical_flip_prob": 1.0,    # Force to happen
            "rotation_prob": 1.0,          # Force to happen
            "rotation_angles": [180],      # Only 180° rotation
            "noise_prob": 1.0,             # Force to happen
            "noise_std": 0.01,
        }
        
        # Create mock dataset instance
        class MockDataset:
            def __init__(self):
                self.augment_config = augment_config
            
            _augment_sequence = UrbanHeatDataset._augment_sequence
        
        dataset = MockDataset()
        
        # Apply augmentation multiple times
        for _ in range(10):
            aug_input, aug_target = dataset._augment_sequence(input_seq.copy(), target_seq.copy())
            
            # Verify shapes are preserved
            assert aug_input.shape == input_seq.shape, \
                f"Input shape changed: expected {input_seq.shape}, got {aug_input.shape}"
            assert aug_target.shape == target_seq.shape, \
                f"Target shape changed: expected {target_seq.shape}, got {aug_target.shape}"

    def test_180_degree_rotation_preserves_dimensions(self):
        """Test that 180° rotation maintains spatial dimensions."""
        # Create mock sequence with non-square dimensions
        input_seq = np.random.randn(4, 5, 1047, 439).astype(np.float32)
        target_seq = np.random.randn(4, 5, 1047, 439).astype(np.float32)
        
        # Only enable rotation
        augment_config = {
            "horizontal_flip_prob": 0.0,
            "vertical_flip_prob": 0.0,
            "rotation_prob": 1.0,
            "rotation_angles": [180],
            "noise_prob": 0.0,
            "noise_std": 0.01,
        }
        
        class MockDataset:
            def __init__(self):
                self.augment_config = augment_config
            
            _augment_sequence = UrbanHeatDataset._augment_sequence
        
        dataset = MockDataset()
        aug_input, aug_target = dataset._augment_sequence(input_seq.copy(), target_seq.copy())
        
        # Verify dimensions are preserved (H, W should not be swapped)
        assert aug_input.shape == (4, 5, 1047, 439), \
            f"180° rotation changed dimensions: got {aug_input.shape}"
        assert aug_target.shape == (4, 5, 1047, 439), \
            f"180° rotation changed dimensions: got {aug_target.shape}"

    def test_batch_compatibility(self):
        """Test that augmented samples can be stacked into batches."""
        # Simulate multiple samples with same original shape
        batch_size = 4
        samples = []
        
        augment_config = {
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.5,
            "rotation_prob": 0.5,
            "rotation_angles": [180],
            "noise_prob": 0.0,
            "noise_std": 0.01,
        }
        
        class MockDataset:
            def __init__(self):
                self.augment_config = augment_config
            
            _augment_sequence = UrbanHeatDataset._augment_sequence
        
        dataset = MockDataset()
        
        # Generate augmented samples
        for _ in range(batch_size):
            input_seq = np.random.randn(4, 5, 1047, 439).astype(np.float32)
            target_seq = np.random.randn(4, 5, 1047, 439).astype(np.float32)
            aug_input, aug_target = dataset._augment_sequence(input_seq, target_seq)
            samples.append((aug_input, aug_target))
        
        # Verify all samples have same shape
        shapes_input = [s[0].shape for s in samples]
        shapes_target = [s[1].shape for s in samples]
        
        assert all(s == shapes_input[0] for s in shapes_input), \
            f"Input shapes are inconsistent: {shapes_input}"
        assert all(s == shapes_target[0] for s in shapes_target), \
            f"Target shapes are inconsistent: {shapes_target}"
        
        # Verify they can be stacked (simulating DataLoader batch creation)
        try:
            batch_input = np.stack([s[0] for s in samples], axis=0)
            batch_target = np.stack([s[1] for s in samples], axis=0)
            assert batch_input.shape == (batch_size, 4, 5, 1047, 439)
            assert batch_target.shape == (batch_size, 4, 5, 1047, 439)
        except ValueError as e:
            raise AssertionError(f"Failed to stack augmented samples into batch: {e}")

    def test_default_config_only_has_180_rotation(self):
        """Test that default config only includes 180° rotation."""
        config = UrbanHeatDataset._default_augment_config()
        
        assert "rotation_angles" in config, "rotation_angles missing from default config"
        assert config["rotation_angles"] == [180], \
            f"Default rotation_angles should be [180], got {config['rotation_angles']}"

    def test_flips_preserve_shape(self):
        """Test that horizontal and vertical flips preserve shape."""
        input_seq = np.random.randn(4, 5, 1047, 439).astype(np.float32)
        target_seq = np.random.randn(4, 5, 1047, 439).astype(np.float32)
        
        # Only enable flips
        augment_config = {
            "horizontal_flip_prob": 1.0,
            "vertical_flip_prob": 1.0,
            "rotation_prob": 0.0,
            "rotation_angles": [180],
            "noise_prob": 0.0,
            "noise_std": 0.01,
        }
        
        class MockDataset:
            def __init__(self):
                self.augment_config = augment_config
            
            _augment_sequence = UrbanHeatDataset._augment_sequence
        
        dataset = MockDataset()
        aug_input, aug_target = dataset._augment_sequence(input_seq.copy(), target_seq.copy())
        
        # Verify shapes are preserved
        assert aug_input.shape == input_seq.shape
        assert aug_target.shape == target_seq.shape


if __name__ == "__main__":
    # Run tests manually
    test = TestAugmentationShapeConsistency()
    
    print("Running test_augmentation_preserves_shape...")
    test.test_augmentation_preserves_shape()
    print("✓ Passed")
    
    print("Running test_180_degree_rotation_preserves_dimensions...")
    test.test_180_degree_rotation_preserves_dimensions()
    print("✓ Passed")
    
    print("Running test_batch_compatibility...")
    test.test_batch_compatibility()
    print("✓ Passed")
    
    print("Running test_default_config_only_has_180_rotation...")
    test.test_default_config_only_has_180_rotation()
    print("✓ Passed")
    
    print("Running test_flips_preserve_shape...")
    test.test_flips_preserve_shape()
    print("✓ Passed")
    
    print("\nAll augmentation tests passed! ✓")
