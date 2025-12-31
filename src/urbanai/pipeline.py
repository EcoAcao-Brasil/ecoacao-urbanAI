"""
UrbanAI Main Pipeline
"""

import logging
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
    Complete pipeline for urban heat prediction.

    Operates data preprocessing, model training, future prediction,
    and intervention analysis.

    Args:
        input_dir: Directory containing raw GeoTIFF files
        output_dir: Directory for all outputs
        config: Path to configuration file or dict
        device: Compute device ('cuda', 'cpu', or 'auto')
        verbose: Enable verbose logging

    Example:
        >>> pipeline = UrbanAIPipeline(
        ...     input_dir="data/raw",
        ...     output_dir="results",
        ...     config="configs/default_config.yaml"
        ... )
        >>> pipeline.run(predict_year=2030)
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

        # Setup logging
        log_file = self.output_dir / "pipeline.log"
        setup_logger(log_file=log_file, verbose=verbose)

        # Load configuration
        self.config = self._load_config(config)

        # Set device
        self.device = self._setup_device(device)

        # Initialize components
        self.preprocessor: Optional[TemporalDataProcessor] = None
        self.trainer: Optional[UrbanAITrainer] = None
        self.predictor: Optional[FuturePredictor] = None
        self.analyzer: Optional[InterventionAnalyzer] = None

        # Results storage
        self._predictions: Optional[Dict[str, Any]] = None
        self._intervention_priorities: Optional[Dict[str, Any]] = None

        logger.info("UrbanAI Pipeline initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")

    def run(
        self,
        preprocess: bool = True,
        train: bool = True,
        predict_year: int = 2030,
        analyze_interventions: bool = True,
        visualize: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute complete pipeline workflow.

        Args:
            preprocess: Run data preprocessing
            train: Train model
            predict_year: Target year for prediction
            analyze_interventions: Perform intervention analysis
            visualize: Generate visualization outputs

        Returns:
            Dictionary containing all results and output paths
        """
        logger.info("Starting UrbanAI Pipeline execution...")

        results = {
            "status": "running",
            "config": self.config,
            "device": str(self.device),
        }

        try:
            # Step 1: Preprocessing
            if preprocess:
                logger.info("Step 1/5: Starting data preprocessing...")
                processed_dir = self._run_preprocessing()
                results["processed_data_dir"] = str(processed_dir)

            # Step 2: Training
            if train:
                logger.info("Step 2/5: Starting model training...")
                model_path = self._run_training()
                results["model_path"] = str(model_path)

            # Step 3: Prediction
            logger.info(f"Step 3/5: Starting future prediction for year {predict_year}...")
            predictions = self._run_prediction(predict_year)
            results["predictions"] = predictions

            # Step 4: Intervention Analysis
            if analyze_interventions:
                logger.info("Step 4/5: Starting intervention analysis...")
                interventions = self._run_analysis(predict_year)
                results["interventions"] = interventions

            # Step 5: Visualization
            if visualize:
                logger.info("Step 5/5: Generating visualizations...")
                viz_outputs = self._generate_visualizations(predict_year)
                results["visualizations"] = viz_outputs

            results["status"] = "success"
            logger.info("Pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
            raise

        return results

    def _run_preprocessing(self) -> Path:
        """Execute preprocessing step."""
        processed_dir = self.output_dir / "processed"

        self.preprocessor = TemporalDataProcessor(
            raw_dir=self.input_dir,
            output_dir=processed_dir,
            config=self.config.get("preprocessing", {}),
        )

        years = self.config.get("preprocessing", {}).get("years")
        if years is None:
            # Generate biennial sequence
            start = self.config["preprocessing"]["start_year"]
            end = self.config["preprocessing"]["end_year"]
            interval = self.config["preprocessing"]["interval"]
            years = list(range(start, end + 1, interval))

        self.preprocessor.process_all_years(
            years=years,
            calculate_indices=True,
            calculate_tocantins=True,
        )

        logger.info(f"Preprocessing complete. Output: {processed_dir}")
        return processed_dir

    def _run_training(self) -> Path:
        """Execute training step."""
        data_dir = self.output_dir / "processed"
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        self.trainer = UrbanAITrainer(
            data_dir=data_dir,
            output_dir=models_dir,
            config=self.config.get("training", {}),
            device=self.device,
        )

        training_results = self.trainer.train(
            epochs=self.config["training"]["epochs"],
            batch_size=self.config["training"]["batch_size"],
        )

        model_path = models_dir / "convlstm_best.pth"
        logger.info(f"Training complete. Best model: {model_path}")
        logger.info(f"Final loss: {training_results['best_loss']:.6f}")

        return model_path

    def _run_prediction(self, target_year: int) -> Dict[str, Any]:
        """Execute prediction step."""
        model_path = self.output_dir / "models" / "convlstm_best.pth"
        predictions_dir = self.output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)

        current_year = self.config["preprocessing"]["end_year"]
        data_dir = self.output_dir / "processed"

        self.predictor = FuturePredictor(
            model_path=model_path,
            data_dir=data_dir,
            output_dir=predictions_dir,
            device=self.device,
        )

        predictions = self.predictor.predict(
            current_year=current_year,
            target_year=target_year,
            save_outputs=True,
        )

        self._predictions = predictions

        logger.info(f"Predictions saved: {predictions['output_path']}")
        return predictions

    def _run_analysis(self, target_year: int) -> Dict[str, Any]:
        """Execute intervention analysis step."""
        current_year = self.config["preprocessing"]["end_year"]
        current_raster = self.output_dir / "processed" / f"{current_year}_features.tif"
        predicted_raster = self._predictions["output_path"]

        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        # Calculate residuals
        residual_calc = ResidualCalculator(
            current_raster=current_raster,
            future_raster=predicted_raster,
            output_dir=analysis_dir,
        )
        residuals = residual_calc.calculate_all_residuals()

        # Identify intervention priorities
        self.analyzer = InterventionAnalyzer(
            residuals_path=residuals["combined_residuals"],
            current_raster=current_raster,
            output_dir=analysis_dir,
        )

        priorities = self.analyzer.identify_priority_zones(
            threshold="high",
            save_geojson=True,
            save_raster=True,
        )

        self._intervention_priorities = priorities

        logger.info(f"Analysis complete. Priority zones: {priorities['n_priority_pixels']}")
        return priorities

    def _generate_visualizations(self, target_year: int) -> Dict[str, Path]:
        """Generate visualization outputs."""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        generator = MapGenerator(output_dir=viz_dir)

        outputs = {}

        # Temporal evolution plot
        data_dir = self.output_dir / "processed"
        outputs["temporal_evolution"] = generator.plot_temporal_evolution(
            data_dir=data_dir,
            metric="LST",
        )

        # Prediction map
        if self._predictions:
            outputs["prediction_map"] = generator.plot_prediction_map(
                prediction_path=self._predictions["output_path"],
                title=f"Predicted Urban Heat Landscape {target_year}",
            )

        # Intervention priority map
        if self._intervention_priorities:
            outputs["intervention_map"] = generator.plot_intervention_map(
                priorities_path=self._intervention_priorities["output_path"],
                title="Urban Heat Intervention Priorities",
            )

        logger.info(f"Visualizations saved: {viz_dir}")
        return outputs

    def get_predictions(self) -> Optional[Dict[str, Any]]:
        """Get prediction results."""
        return self._predictions

    def get_intervention_priorities(self) -> Optional[Dict[str, Any]]:
        """Get intervention priority analysis results."""
        return self._intervention_priorities

    @staticmethod
    def _load_config(config: Optional[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Any]:
        """Load configuration from file or dict."""
        if config is None:
            return UrbanAIPipeline._default_config()

        if isinstance(config, dict):
            return config

        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "preprocessing": {
                "start_year": 1985,
                "end_year": 2025,
                "interval": 2,
                "season": "07-01_12-31",
                "calculate_indices": True,
                "calculate_tocantins": True,
            },
            "training": {
                "epochs": 100,
                "batch_size": 8,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "early_stopping": {"patience": 15, "min_delta": 0.0001},
            },
            "model": {
                "architecture": "convlstm",
                "input_channels": 7,
                "hidden_dims": [64, 128, 256, 256, 128, 64],
                "kernel_size": 3,
                "num_layers": 6,
            },
        }

    @staticmethod
    def _setup_device(device: str) -> str:
        """Setup compute device."""
        import torch

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
