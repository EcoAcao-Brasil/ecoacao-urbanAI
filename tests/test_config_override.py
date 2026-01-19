"""
Test configuration override behavior for UrbanAI.

Ensures that user-provided config properly overrides internal defaults.
"""

import tempfile
from pathlib import Path

from urbanai.training.trainer import UrbanAITrainer


def test_sequence_length_override_nested():
    """Ensure user-provided sequence_length in nested training key overrides defaults"""
    config = {
        'training': {
            'sequence_length': 4
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=config
        )
        
        # Should respect user config in nested training key
        assert trainer.config['training']['sequence_length'] == 4
        
        # Should NOT use hardcoded default of 10
        assert trainer.config['training']['sequence_length'] != 10


def test_sequence_length_override_top_level():
    """Ensure user-provided sequence_length at top-level overrides defaults"""
    config = {
        'sequence_length': 7
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=config
        )
        
        # Should respect user config at top level
        assert trainer.config['sequence_length'] == 7
        
        # Should NOT use hardcoded default of 10
        assert trainer.config['sequence_length'] != 10


def test_nested_config_priority():
    """Nested training.* config should take priority over top-level"""
    config = {
        'sequence_length': 5,  # Top-level
        'training': {
            'sequence_length': 8  # Nested - should win
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=config
        )
        
        # Nested value should be preserved
        assert trainer.config['training']['sequence_length'] == 8
        # Top-level value should also be preserved
        assert trainer.config['sequence_length'] == 5


def test_deep_merge_preserves_defaults():
    """Ensure partial user config merges with defaults rather than replacing them"""
    config = {
        'training': {
            'sequence_length': 4,
            'epochs': 150
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=config
        )
        
        # User values should be present
        assert trainer.config['training']['sequence_length'] == 4
        
        # Default values not specified by user should still be present
        assert 'normalization_method' in trainer.config['training']
        assert trainer.config['training']['normalization_method'] == 'zscore'
        
        # Model config should be completely preserved from defaults
        assert 'model' in trainer.config
        assert trainer.config['model']['input_channels'] == 5


def test_config_merge_overwrites_not_appends():
    """Ensure user config values overwrite defaults, not append to them"""
    config = {
        'model': {
            'hidden_dims': [32, 64, 128]  # Different from default
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=config
        )
        
        # Should use user's hidden_dims, not default
        assert trainer.config['model']['hidden_dims'] == [32, 64, 128]
        assert trainer.config['model']['hidden_dims'] != [64, 128, 256, 256, 128, 64]


def test_empty_config_uses_all_defaults():
    """Ensure that passing no config uses all defaults"""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UrbanAITrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            config=None
        )
        
        # Should have all default values
        assert trainer.config['sequence_length'] == 10
        assert trainer.config['prediction_horizon'] == 1
        assert trainer.config['learning_rate'] == 0.001
        assert trainer.config['training']['normalization_method'] == 'zscore'


if __name__ == '__main__':
    # Run tests manually
    print("Running test_sequence_length_override_nested...")
    test_sequence_length_override_nested()
    print("✓ Passed")
    
    print("Running test_sequence_length_override_top_level...")
    test_sequence_length_override_top_level()
    print("✓ Passed")
    
    print("Running test_nested_config_priority...")
    test_nested_config_priority()
    print("✓ Passed")
    
    print("Running test_deep_merge_preserves_defaults...")
    test_deep_merge_preserves_defaults()
    print("✓ Passed")
    
    print("Running test_config_merge_overwrites_not_appends...")
    test_config_merge_overwrites_not_appends()
    print("✓ Passed")
    
    print("Running test_empty_config_uses_all_defaults...")
    test_empty_config_uses_all_defaults()
    print("✓ Passed")
    
    print("\nAll tests passed! ✓")
