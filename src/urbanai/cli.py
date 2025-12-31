"""
Command Line Interface for UrbanAI

Provides CLI commands for running UrbanAI workflows.
"""

import argparse
import logging
import sys
from pathlib import Path

from urbanai import __version__
from urbanai.pipeline import UrbanAIPipeline


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UrbanAI - Urban Heat Island Prediction Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"UrbanAI {__version__}",
    )
    
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
    predict_parser = subparsers.add_parser("predict", help="Predict future only")
    predict_parser.add_argument("--model", required=True, help="Model checkpoint path")
    predict_parser.add_argument("--data", required=True, help="Data directory")
    predict_parser.add_argument("--year", type=int, required=True, help="Year to predict")
    predict_parser.add_argument("--output", required=True, help="Output directory")
    predict_parser.add_argument("--device", default="auto", help="Device")
    
    # Analyze only
    analyze_parser = subparsers.add_parser("analyze", help="Analyze predictions")
    analyze_parser.add_argument("--current", required=True, help="Current year raster")
    analyze_parser.add_argument("--predicted", required=True, help="Predicted raster")
    analyze_parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
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
        elif args.command == "analyze":
            run_analyze(args)
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
        verbose=getattr(args, 'verbose', False),
    )
    
    results = pipeline.run(predict_year=args.predict_year)
    
    if results["status"] == "success":
        print("\nPipeline completed successfully.")
        print(f"Results saved to: {args.output}")
    else:
        print("\n Pipeline failed")
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
    print(f"\n Preprocessing complete: {len(results)} years processed")


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
    
    print(f"\n Training complete")
    print(f"Best validation loss: {results['best_loss']:.6f}")


def run_predict(args) -> None:
    """Run prediction only."""
    from urbanai.prediction import FuturePredictor
    
    predictor = FuturePredictor(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        device=args.device,
    )
    
    results = predictor.predict(
        current_year=2025,
        target_year=args.year,
        save_outputs=True,
    )
    
    print(f"\n Prediction complete for {args.year}")
    print(f"Output: {results['output_path']}")


def run_analyze(args) -> None:
    """Run analysis only."""
    from urbanai.analysis import InterventionAnalyzer, ResidualCalculator
    
    # Calculate residuals
    residual_calc = ResidualCalculator(
        current_raster=args.current,
        future_raster=args.predicted,
        output_dir=args.output,
    )
    
    residuals = residual_calc.calculate_all_residuals()
    
    # Analyze interventions
    analyzer = InterventionAnalyzer(
        residuals_path=residuals["combined_residuals"],
        current_raster=args.current,
        output_dir=args.output,
    )
    
    priorities = analyzer.identify_priority_zones(save_geojson=True)
    
    print(f"\n Analysis complete")
    print(f"Priority zones: {priorities['n_hotspot_zones']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
