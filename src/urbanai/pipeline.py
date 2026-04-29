"""
UrbanAI Pipeline

Orchestrates the full workflow: raw GeoTIFF preprocessing → model training → predicted raster output.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from urbanai.prediction import FuturePredictor
from urbanai.preprocessing import TemporalDataProcessor
from urbanai.training import UrbanAITrainer
from urbanai.utils.config import get_input_channels
from urbanai.utils.logging import setup_logger
from urbanai.visualization import MapGenerator

logger = logging.getLogger(__name__)


class UrbanAIPipeline:
    """
    Runs preprocessing, training, and prediction in sequence.

    Each stage can be toggled independently, allowing the pipeline to resume
    from a saved model or preprocessed data without repeating earlier steps.
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        device: str = "auto",
        verbose: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        setup_logger(log_file=self.output_dir / "pipeline.log", verbose=verbose)

        self.config = self._load_config(config)

        if "model" not in self.config:
            self.config["model"] = {}

        if "input_channels" not in self.config.get("model", {}):
            self.config["model"]["input_channels"] = get_input_channels(self.config)
            self.config["model"]["output_channels"] = get_input_channels(self.config)

        self.device = self._setup_device(device)

        self._predictions: Optional[Dict[str, Any]] = None

        logger.info("UrbanAI Pipeline initialized")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model channels: {self.config['model']['input_channels']}")

    def run(
        self,
        preprocess: bool = True,
        train: bool = True,
        predict_year: Optional[int] = None,
        visualize: bool = True,
    ) -> Dict[str, Any]:
        """
        Run pipeline stages in sequence.

        Args:
            preprocess: Run preprocessing on raw GeoTIFFs.
            train: Train the ConvLSTM model.
            predict_year: Future year to predict. Must exceed the most recent data year.
            visualize: Generate visualization outputs (PNG maps).

        Returns:
            Dictionary with status and paths to all outputs.
        """
        logger.info("Starting UrbanAI Pipeline...")

        results: Dict[str, Any] = {
            "status": "running",
            "config": self.config,
            "device": str(self.device),
        }

        try:
            # 1. Preprocess
            if preprocess:
                logger.info("Step 1/4: Preprocessing data...")
                processed_dir = self._run_preprocessing()
                results["processed_data_dir"] = str(processed_dir)
            else:
                processed_dir = self.output_dir / "processed"

            # 2. Train
            if train:
                logger.info("Step 2/4: Training model...")
                model_path = self._run_training()
                results["model_path"] = str(model_path)
            else:
                model_path = self.output_dir / "models" / "convlstm_best.pth"

            # 3. Predict
            if predict_year is not None:
                logger.info(f"Step 3/4: Predicting year {predict_year}...")
                predictions = self._run_prediction(predict_year)
                results["predictions"] = predictions
            else:
                logger.info("Step 3/4: Skipping prediction (no target year specified)")

            # 4. Visualize
            if visualize and predict_year is not None:
                logger.info("Step 4/4: Generating visualizations...")
                viz_outputs = self._generate_visualizations(predict_year)
                results["visualizations"] = viz_outputs
            else:
                logger.info("Step 4/4: Skipping visualization")

            results["status"] = "success"
            logger.info("Pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
            raise

        return results

    def _run_preprocessing(self) -> Path:
        """Normalize raw GeoTIFFs and calculate spectral indices."""
        processed_dir = self.output_dir / "processed"

        processor = TemporalDataProcessor(
            raw_dir=self.input_dir,
            output_dir=processed_dir,
            config=self.config.get("preprocessing", {}),
        )

        years = self.config.get("preprocessing", {}).get("years")
        if years is None:
            start = self.config["preprocessing"]["start_year"]
            end = self.config["preprocessing"]["end_year"]
            interval = self.config["preprocessing"]["interval"]
            years = list(range(start, end + 1, interval))

        processor.process_all_years(
            years=years,
            calculate_indices=True,
            calculate_tocantins=self.config["preprocessing"]["tocantins"].get("enabled", False),
        )

        logger.info(f"Preprocessing complete: {processed_dir}")
        return processed_dir

    def _run_training(self) -> Path:
        """Train the ConvLSTM model on preprocessed data."""
        data_dir = self.output_dir / "processed"
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        trainer = UrbanAITrainer(
            data_dir=data_dir,
            output_dir=models_dir,
            config=self.config,
            device=self.device,
        )

        training_results = trainer.train(
            epochs=self.config["training"]["epochs"],
            batch_size=self.config["training"]["batch_size"],
        )

        model_path = models_dir / "convlstm_best.pth"
        logger.info(f"Training complete: {model_path}")
        logger.info(f"Best validation loss: {training_results['best_loss']:.6f}")

        return model_path

    def _run_prediction(self, target_year: int) -> Dict[str, Any]:
        """Generate a predicted raster for target_year using the trained model."""
        model_path = self.output_dir / "models" / "convlstm_best.pth"
        predictions_dir = self.output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)

        processed_dir = self.output_dir / "processed"
        current_year = self._get_most_recent_year(processed_dir)

        if current_year is None:
            raise ValueError(f"No processed files found in {processed_dir}")

        if target_year <= current_year:
            raise ValueError(
                f"Target year ({target_year}) must be greater than the most recent "
                f"data year ({current_year})"
            )

        logger.info(f"Predicting from {current_year} to {target_year}")

        predictor = FuturePredictor(
            model_path=model_path,
            data_dir=processed_dir,
            output_dir=predictions_dir,
            device=self.device,
        )

        predictions = predictor.predict(
            current_year=current_year,
            target_year=target_year,
            save_outputs=True,
        )

        self._predictions = predictions
        logger.info(f"Prediction saved: {predictions['output_path']}")
        return predictions

    def _generate_visualizations(self, target_year: int) -> Dict[str, Path]:
        """Generate PNG maps for the processed data and prediction."""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        generator = MapGenerator(output_dir=viz_dir)
        outputs = {}

        data_dir = self.output_dir / "processed"
        outputs["temporal_evolution"] = generator.plot_temporal_evolution(
            data_dir=data_dir,
            metric="LST",
        )

        if self._predictions:
            outputs["prediction_map"] = generator.plot_prediction_map(
                prediction_path=self._predictions["output_path"],
                title=f"Predicted Urban Heat {target_year}",
            )

        logger.info(f"Visualizations saved: {viz_dir}")
        return outputs

    def _get_most_recent_year(self, data_dir: Path) -> Optional[int]:
        """Return the most recent year found in processed feature files."""
        files = sorted(data_dir.glob("*_features*.tif"))
        if not files:
            return None
        return max(self._extract_year(f.name) for f in files)

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract a 4-digit year from a filename."""
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")

    def get_predictions(self) -> Optional[Dict[str, Any]]:
        """Return the last prediction result, or None if no prediction has been run."""
        return self._predictions

    @staticmethod
    def _load_config(config: Optional[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Any]:
        """Load configuration from a file path, a dict, or built-in defaults."""
        if config is None:
            return UrbanAIPipeline._default_config()

        if isinstance(config, dict):
            return config

        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Built-in defaults. Override any key via a config file or dict."""
        return {
            "preprocessing": {
                "start_year": 1985,
                "end_year": 2023,
                "interval": 2,
                "tocantins": {
                    "enabled": False,
                },
            },
            "training": {
                "epochs": 100,
                "batch_size": 8,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "early_stopping": {"patience": 15},
            },
            "model": {
                "input_channels": 5,
                "hidden_dims": [64, 128, 256, 256, 128, 64],
                "kernel_size": 3,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
            },
        }

    @staticmethod
    def _setup_device(device: str) -> str:
        import torch

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
