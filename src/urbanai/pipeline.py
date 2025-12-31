"""
Main pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional
import yaml

from .data_loader import DataLoader
from .model import ConvLSTMPredictor
from .trainer import ModelTrainer
from .predictor import FuturePredictor

logger = logging.getLogger(__name__)


class UrbanAIPipeline:
    """
    Complete pipeline for urban heat prediction.
    
    Usage:
        pipeline = UrbanAIPipeline("configs/config.yaml")
        pipeline.run_all()
    """
    
    def __init__(self, config_path: str):
        """Initialize with config file."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['data']['processed_dir'])
        self.output_dir = Path(self.config['output']['results_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Pipeline initialized")
    
    def run_all(self):
        """Run complete pipeline: process → train → predict."""
        logger.info("Starting complete pipeline")
        
        # Step 1: Process data
        self.process_data()
        
        # Step 2: Train model
        self.train_model()
        
        # Step 3: Generate predictions
        self.predict_future()
        
        logger.info("Pipeline complete!")
    
    def process_data(self):
        """Process raw Landsat data."""
        logger.info("Step 1: Processing data")
        
        loader = DataLoader(
            raw_dir=self.config['data']['raw_dir'],
            output_dir=self.config['data']['processed_dir'],
            years=self.config['data']['years']
        )
        
        loader.process_all_years()
    
    def train_model(self):
        """Train ConvLSTM model."""
        logger.info("Step 2: Training AI model")
        
        trainer = ModelTrainer(
            data_dir=self.data_dir,
            config=self.config['model']
        )
        
        self.model = trainer.train()
        
        # Save model
        model_path = self.output_dir / "model.pth"
        trainer.save_model(model_path)
        logger.info(f"Model saved: {model_path}")
    
    def predict_future(self):
        """Generate future predictions."""
        logger.info("Step 3: Generating predictions")
        
        predictor = FuturePredictor(
            model=self.model,
            data_dir=self.data_dir,
            config=self.config['prediction']
        )
        
        for year in self.config['prediction']['target_years']:
            prediction = predictor.predict_year(year)
            
            # Save prediction
            output_path = self.output_dir / f"prediction_{year}.tif"
            predictor.save_prediction(prediction, output_path)
            
            # Calculate priorities
            priorities = predictor.calculate_priorities(prediction)
            priority_path = self.output_dir / f"priorities_{year}.tif"
            predictor.save_priorities(priorities, priority_path)
            
            logger.info(f"Saved predictions for {year}")
