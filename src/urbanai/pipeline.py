"""
UrbanAI Main Pipeline

Operates the full workflow from raw GeoTIFF to urban heat prediction and intervention analysis.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from urbanai.analysis import InterventionAnalyzer, ResidualCalculator
from urbanai.prediction import FuturePredictor
from urbanai.preprocessing import TemporalDataProcessor
from urbanai.training import UrbanAITrainer
from urbanai.utils.logging import setup_logger
from urbanai.visualization import MapGenerator

logger = logging.getLogger(__name__)


class UrbanAIPipeline:
    """
    Main entry point for the Urban Heat Island assessment system.
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
        
        # Ensure we have a place to dump logs and results immediately
        self.output_dir.mkdir(parents=True, exist_ok=True)

        setup_logger(log_file=self.output_dir / "pipeline.log", verbose=verbose)

        self.config = self._load_config(config)
        self.device = self._setup_device(device)

        # State tracking for results that are passed between steps
        self._predictions: Optional[Dict[str, Any]] = None
        self._intervention_priorities: Optional[Dict[str, Any]] = None

        logger.info("UrbanAI Pipeline initialized")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Device: {self.device}")

    def run(
        self,
        preprocess: bool = True,
        train: bool = True,
        predict_year: Optional[int] = None,
        analyze_interventions: bool = True,
        visualize: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs the pipeline stages in sequence.
        
        You can toggle specific stages off (e.g., set train=False to use a saved model).
        Note: `predict_year` is mandatory if you want to generate new forecasts or analysis.
        """
        logger.info("Starting UrbanAI Pipeline...")

        results = {
            "status": "running",
            "config": self.config,
            "device": str(self.device),
        }

        try:
            # 1. Prepare Data
            if preprocess:
                logger.info("Step 1/5: Preprocessing data...")
                processed_dir = self._run_preprocessing()
                results["processed_data_dir"] = str(processed_dir)
            else:
                # Assume data exists if we aren't running preprocessing
                processed_dir = self.output_dir / "processed"

            # 2. Model Training
            if train:
                logger.info("Step 2/5: Training model...")
                model_path = self._run_training()
                results["model_path"] = str(model_path)
            else:
                # Fallback to a previously trained best model
                model_path = self.output_dir / "models" / "convlstm_best.pth"

            # 3. Future Prediction
            if predict_year is not None:
                logger.info(f"Step 3/5: Predicting year {predict_year}...")
                predictions = self._run_prediction(predict_year)
                results["predictions"] = predictions
            else:
                logger.info("Step 3/5: Skipping prediction (no target year specified)")
                predictions = None

            # 4. Intervention Analysis (Residuals & Hotspots)
            if analyze_interventions and predict_year is not None:
                logger.info("Step 4/5: Analyzing interventions...")
                interventions = self._run_analysis(predict_year)
                results["interventions"] = interventions
            else:
                logger.info("Step 4/5: Skipping analysis")
                interventions = None

            # 5. Visualization Generation
            if visualize and predict_year is not None:
                logger.info("Step 5/5: Generating visualizations...")
                viz_outputs = self._generate_visualizations(predict_year)
                results["visualizations"] = viz_outputs
            else:
                logger.info("Step 5/5: Skipping visualization")

            results["status"] = "success"
            logger.info("Pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
            raise

        return results

    def _run_preprocessing(self) -> Path:
        """Handles the normalization and index calculation for raw GeoTIFFs."""
        processed_dir = self.output_dir / "processed"

        processor = TemporalDataProcessor(
            raw_dir=self.input_dir,
            output_dir=processed_dir,
            config=self.config.get("preprocessing", {}),
        )

        # If specific years aren't listed in config, generate the range automatically
        years = self.config.get("preprocessing", {}).get("years")
        if years is None:
            start = self.config["preprocessing"]["start_year"]
            end = self.config["preprocessing"]["end_year"]
            interval = self.config["preprocessing"]["interval"]
            years = list(range(start, end + 1, interval))

        processor.process_all_years(
            years=years,
            calculate_indices=True,
            calculate_tocantins=True,
        )

        logger.info(f"Preprocessing complete: {processed_dir}")
        return processed_dir

    def _run_training(self) -> Path:
        """Initializes the trainer and runs the training loop."""
        data_dir = self.output_dir / "processed"
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        trainer = UrbanAITrainer(
            data_dir=data_dir,
            output_dir=models_dir,
            config=self.config.get("training", {}),
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
        """Loads the best model and generates a forecast for the target year."""
        model_path = self.output_dir / "models" / "convlstm_best.pth"
        predictions_dir = self.output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)

        processed_dir = self.output_dir / "processed"
        
        # Dynamically determine the most recent year from available data
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
        logger.info(f"Predictions saved: {predictions['output_path']}")
        return predictions

    def _run_analysis(self, target_year: int) -> Dict[str, Any]:
        """Compares current state vs. predicted state to identify priority zones."""
        processed_dir = self.output_dir / "processed"
        
        # Dynamically determine the most recent year
        current_year = self._get_most_recent_year(processed_dir)
        
        if current_year is None:
            raise ValueError(f"No processed files found in {processed_dir}")

        current_raster = processed_dir / f"{current_year}_features_complete.tif"
        
        # Handle alternative naming patterns
        if not current_raster.exists():
            current_raster = processed_dir / f"{current_year}_features.tif"
            
        if not current_raster.exists():
            raise FileNotFoundError(
                f"Could not find raster for year {current_year} in {processed_dir}"
            )
        
        predicted_raster = self._predictions["output_path"]

        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        # 1. Calculate the 'residual' (difference between prediction and current state)
        residual_calc = ResidualCalculator(
            current_raster=current_raster,
            future_raster=predicted_raster,
            output_dir=analysis_dir,
            weights=self.config.get("analysis", {}).get("priority_weights"),
        )
        residuals = residual_calc.calculate_all_residuals()

        # 2. Determine where interventions are needed most
        analyzer = InterventionAnalyzer(
            residuals_path=residuals["combined_residuals"],
            current_raster=current_raster,
            output_dir=analysis_dir,
        )

        priorities = analyzer.identify_priority_zones(
            threshold=self.config.get("analysis", {}).get("threshold", "high"),
            save_geojson=True,
            save_raster=True,
        )

        self._intervention_priorities = priorities
        logger.info(f"Analysis complete: {priorities['n_hotspot_zones']} priority zones")
        return priorities

    def _generate_visualizations(self, target_year: int) -> Dict[str, Path]:
        """Creates plotting outputs (maps and temporal charts)."""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        generator = MapGenerator(output_dir=viz_dir)

        outputs = {}

        # Historical trend
        data_dir = self.output_dir / "processed"
        outputs["temporal_evolution"] = generator.plot_temporal_evolution(
            data_dir=data_dir,
            metric="LST",
        )

        # Future Heat Map
        if self._predictions:
            outputs["prediction_map"] = generator.plot_prediction_map(
                prediction_path=self._predictions["output_path"],
                title=f"Predicted Urban Heat {target_year}",
            )

        # Priority Zones Map
        if self._intervention_priorities:
            outputs["intervention_map"] = generator.plot_intervention_map(
                priorities_path=self._intervention_priorities["output_path"],
                title="Intervention Priorities",
            )

        logger.info(f"Visualizations saved: {viz_dir}")
        return outputs

    def _get_most_recent_year(self, data_dir: Path) -> Optional[int]:
        """
        Dynamically determine the most recent year from available processed files.
        
        Args:
            data_dir: Directory containing processed feature files
            
        Returns:
            Most recent year, or None if no files found
        """
        files = sorted(data_dir.glob("*_features*.tif"))
        
        if not files:
            return None
        
        years = sorted([self._extract_year(f.name) for f in files])
        return max(years)

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract year from filename."""
        import re
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")

    def get_predictions(self) -> Optional[Dict[str, Any]]:
        return self._predictions

    def get_intervention_priorities(self) -> Optional[Dict[str, Any]]:
        return self._intervention_priorities

    @staticmethod
    def _load_config(config: Optional[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Any]:
        """Resolves configuration from a file path, a dictionary, or defaults."""
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
        """
        Provides sensible defaults for standard ConvLSTM heat prediction.
        
        Note: These defaults use 1985-2023 as an example range. Users should
        update these values based on their specific data availability.
        """
        return {
            "preprocessing": {
                "start_year": 1985,
                "end_year": 2023,  # Changed from 2025 to avoid assumptions
                "interval": 2,
            },
            "training": {
                "epochs": 100,
                "batch_size": 8,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "early_stopping": {"patience": 15},
            },
            "model": {
                "input_channels": 7,
                "hidden_dims": [64, 128, 256, 256, 128, 64],
                "kernel_size": 3,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
            },
            "analysis": {
                "threshold": "high",
            },
        }

    @staticmethod
    def _setup_device(device: str) -> str:
        import torch

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
