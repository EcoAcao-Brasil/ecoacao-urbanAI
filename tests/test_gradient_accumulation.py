"""
Test gradient accumulation functionality for UrbanAI Trainer.

Ensures that gradient accumulation works correctly with different configurations.
"""

import tempfile
from pathlib import Path

from urbanai.training.trainer import UrbanAITrainer


def test_gradient_accumulation_default():
    """Ensure gradient_accumulation_steps defaults to 1 (no accumulation)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=None,
        )

        # Should default to 1 (no accumulation)
        assert trainer.config["gradient_accumulation_steps"] == 1


def test_gradient_accumulation_custom():
    """Ensure user-provided gradient_accumulation_steps is respected"""
    config = {"gradient_accumulation_steps": 4}

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=config,
        )

        # Should use user value
        assert trainer.config["gradient_accumulation_steps"] == 4


def test_gradient_accumulation_nested():
    """Ensure gradient_accumulation_steps works in nested training config"""
    config = {"training": {"gradient_accumulation_steps": 8}}

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=config,
        )

        # Should respect nested value via deep merge
        assert trainer.config["training"]["gradient_accumulation_steps"] == 8
        # Top-level default should still exist
        assert trainer.config["gradient_accumulation_steps"] == 1


def test_gradient_accumulation_override():
    """Ensure user config overrides default"""
    config = {
        "gradient_accumulation_steps": 2,
        "training": {"batch_size": 4},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=config,
        )

        # User value should override default
        assert trainer.config["gradient_accumulation_steps"] == 2
        # Should NOT use default of 1
        assert trainer.config["gradient_accumulation_steps"] != 1


def test_gradient_accumulation_with_batch_size():
    """Test realistic configuration with batch_size and gradient accumulation"""
    config = {
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=config,
        )

        # Both values should be preserved
        assert trainer.config["training"]["batch_size"] == 2
        assert trainer.config["training"]["gradient_accumulation_steps"] == 4


def test_gradient_accumulation_preserves_other_config():
    """Ensure gradient accumulation doesn't affect other configuration"""
    config = {
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "optimizer": "adamw",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=config,
        )

        # Gradient accumulation should be set
        assert trainer.config["gradient_accumulation_steps"] == 4
        # Other config should be preserved
        assert trainer.config["learning_rate"] == 0.0001
        assert trainer.config["optimizer"] == "adamw"
        # Defaults should still be present
        assert trainer.config["gradient_clip"] == 1.0


if __name__ == "__main__":
    # Run tests manually
    print("Running test_gradient_accumulation_default...")
    test_gradient_accumulation_default()
    print("✓ Passed")

    print("Running test_gradient_accumulation_custom...")
    test_gradient_accumulation_custom()
    print("✓ Passed")

    print("Running test_gradient_accumulation_nested...")
    test_gradient_accumulation_nested()
    print("✓ Passed")

    print("Running test_gradient_accumulation_override...")
    test_gradient_accumulation_override()
    print("✓ Passed")

    print("Running test_gradient_accumulation_with_batch_size...")
    test_gradient_accumulation_with_batch_size()
    print("✓ Passed")

    print("Running test_gradient_accumulation_preserves_other_config...")
    test_gradient_accumulation_preserves_other_config()
    print("✓ Passed")

    print("\nAll gradient accumulation tests passed! ✓")
