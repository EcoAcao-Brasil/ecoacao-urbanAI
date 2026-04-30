"""
Command Line Interface for UrbanAI
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from urbanai import __version__
from urbanai.pipeline import UrbanAIPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UrbanAI - Spatiotemporal Urban Heat Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"UrbanAI {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run complete pipeline
    run_parser = subparsers.add_parser("run", help="Run complete pipeline")
    run_parser.add_argument("--input", required=True, help="Input data directory")
    run_parser.add_argument("--output", required=True, help="Output directory")
    run_parser.add_argument("--config", help="Configuration file (YAML)")
    run_parser.add_argument("--predict-year", type=int, default=2030, help="Year to predict")
    run_parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    run_parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Preprocess only
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data only")
    preprocess_parser.add_argument("--input", required=True, help="Raw data directory")
    preprocess_parser.add_argument("--output", required=True, help="Output directory")
    preprocess_parser.add_argument("--config", help="Configuration file")

    # Train only
    train_parser = subparsers.add_parser("train", help="Train model only")
    train_parser.add_argument("--data", required=True, help="Processed data directory")
    train_parser.add_argument("--output", required=True, help="Model output directory")
    train_parser.add_argument("--config", help="Configuration file")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--device", default="auto", help="Device")

    # Predict only
    predict_parser = subparsers.add_parser("predict", help="Generate predicted raster")
    predict_parser.add_argument("--model", required=True, help="Model checkpoint path")
    predict_parser.add_argument("--data", required=True, help="Processed data directory")
    predict_parser.add_argument("--year", type=int, required=True, help="Year to predict")
    predict_parser.add_argument("--output", required=True, help="Output directory")
    predict_parser.add_argument("--device", default="auto", help="Device")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    try:
        if args.command == "run":
            run_pipeline(args)
        elif args.command == "preprocess":
            run_preprocess(args)
        elif args.command == "train":
            run_train(args)
        elif args.command == "predict":
            run_predict(args)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)


def run_pipeline(args) -> None:
    """Run complete pipeline."""
    pipeline = UrbanAIPipeline(
        input_dir=args.input,
        output_dir=args.output,
        config=args.config,
        device=args.device,
        verbose=getattr(args, "verbose", False),
    )

    results = pipeline.run(predict_year=args.predict_year)

    if results["status"] == "success":
        logger.info(f"Pipeline completed. Results: {args.output}")
    else:
        logger.error("Pipeline failed")
        sys.exit(1)


def run_preprocess(args) -> None:
    """Run preprocessing only."""
    from urbanai.preprocessing import TemporalDataProcessor

    processor = TemporalDataProcessor(
        raw_dir=args.input,
        output_dir=args.output,
        config=args.config,
    )

    results = processor.process_all_years()
    logger.info(f"Preprocessing complete: {len(results)} years processed")


def run_train(args) -> None:
    """Run training only."""
    from urbanai.training import UrbanAITrainer

    trainer = UrbanAITrainer(
        data_dir=args.data,
        output_dir=args.output,
        config=args.config,
        device=args.device,
    )

    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    logger.info(f"Training complete. Best validation loss: {results['best_loss']:.6f}")


def run_predict(args) -> None:
    """Generate a predicted raster for a future year."""
    from urbanai.prediction import FuturePredictor

    data_dir = Path(args.data)
    current_year = _get_most_recent_year(data_dir)

    if current_year is None:
        raise ValueError(f"No processed files found in {data_dir}")

    if args.year <= current_year:
        raise ValueError(
            f"Target year ({args.year}) must be greater than the most recent "
            f"data year ({current_year})"
        )

    logger.info(f"Predicting from {current_year} to {args.year}")

    predictor = FuturePredictor(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        device=args.device,
    )

    results = predictor.predict(
        current_year=current_year,
        target_year=args.year,
        save_outputs=True,
    )

    logger.info(f"Prediction complete: {results['output_path']}")


def _get_most_recent_year(data_dir: Path) -> Optional[int]:
    """Return the most recent year from processed feature files in data_dir."""
    files = sorted(data_dir.glob("*_features*.tif"))
    if not files:
        return None
    years = [_extract_year(f.name) for f in files]
    return max(years)


def _extract_year(filename: str) -> int:
    """Extract a 4-digit year from a filename."""
    match = re.search(r"(\d{4})", filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract year from: {filename}")


if __name__ == "__main__":
    main()
