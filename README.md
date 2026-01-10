# UrbanAI, by EcoAção Brasil (Brazil EcoAction)

**Deep learning framework for spatiotemporal urban heat prediction using ConvLSTM and satellite imagery**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/urbanai.svg)](https://badge.fury.io/py/urbanai)

Developed by [EcoAção Brasil](https://ecoacaobrasil.org)

---

## Overview

UrbanAI is a comprehensive deep learning framework for assessing the evolution of the Urban Heat Island (UHI) phenomenon with an emphasis on urban climate planning. Built on ConvLSTM architecture, it analyzes historical Landsat data to forecast future thermal landscapes and identify priority areas for urban climate intervention.

**Key Feature**: UrbanAI is fully modular and does not assume any specific temporal range - you define your own data years, prediction targets, and analysis periods based on your available satellite imagery.

### Key Features

- **Flexible Temporal Configuration**: Works with any date range from your Landsat data (not limited to 1985-2025)
- **Spatiotemporal Deep Learning**: ConvLSTM-based architecture for temporal pattern learning
- **Multi-metric Prediction**: LST, NDBI, NDVI, NDWI, NDBSI, and intra-urban thermal anomaly metrics (Severity Score and Impact Score)
- **Tocantins Integration**: Integration with the [Tocantins Framework](https://github.com/EcoAcao-Brasil/tocantins-framework) for Impact Score (IS) and Severity Score (SS) calculation
- **Intervention Prioritization**: Automated identification of critical urban heat mitigation zones
- **Production Ready**: Designed for research and platform integration
- **Scalable Architecture**: GPU-optimized, batch processing, cloud-deployment ready

---

## Installation

### From PyPI (Recommended)

```bash
pip install urbanai
```

### From Source

```bash
git clone https://github.com/EcoAcao-Brasil/ecoacao-urbanai
cd ecoacao-urbanai
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
pre-commit install
```

### Google Colab

```python
!pip install urbanai tocantins-framework
```

---

## Quick Start

### Basic Usage with Custom Years

```python
from urbanai import UrbanAIPipeline

# Initialize pipeline with your specific temporal range
pipeline = UrbanAIPipeline(
    input_dir="data/raw",
    output_dir="results",
    config={
        "preprocessing": {
            "start_year": 2000,  # Your earliest data year
            "end_year": 2022,    # Your most recent data year
            "interval": 2        # Biennial or annual (1)
        },
        "training": {
            "epochs": 100,
            "batch_size": 8
        }
    }
)

# Run complete workflow - predict any future year
pipeline.run(
    preprocess=True,
    train=True,
    predict_year=2030,  # Any year beyond your end_year
    analyze_interventions=True
)

# Access results
predictions = pipeline.get_predictions()
intervention_map = pipeline.get_intervention_priorities()
```

### Example: Different Temporal Scenarios

```python
# Scenario 1: Long-term historical analysis (40 years)
config_historical = {
    "preprocessing": {
        "start_year": 1985,
        "end_year": 2023,
        "interval": 2
    }
}
pipeline = UrbanAIPipeline(input_dir="data/raw", output_dir="results", config=config_historical)
pipeline.run(predict_year=2035)  # 12-year prediction

# Scenario 2: Recent high-resolution analysis (annual data)
config_recent = {
    "preprocessing": {
        "start_year": 2015,
        "end_year": 2023,
        "interval": 1  # Annual instead of biennial
    },
    "training": {
        "sequence_length": 8  # Adjust based on your interval
    }
}
pipeline = UrbanAIPipeline(input_dir="data/raw", output_dir="results", config=config_recent)
pipeline.run(predict_year=2028)  # 5-year prediction

# Scenario 3: Mid-term focused study
config_midterm = {
    "preprocessing": {
        "start_year": 2010,
        "end_year": 2020,
        "interval": 2
    }
}
pipeline = UrbanAIPipeline(input_dir="data/raw", output_dir="results", config=config_midterm)
pipeline.run(predict_year=2025)
```

### Step-by-Step Workflow

```python
from urbanai import preprocessing, models, training, prediction, analysis

# 1. Prepare data with your specific years
processor = preprocessing.TemporalDataProcessor(
    raw_dir="data/raw",
    output_dir="data/processed"
)

# Define YOUR years based on available Landsat data
my_years = list(range(2000, 2023, 2))  # 2000, 2002, ..., 2022

processor.process_all_years(
    years=my_years,
    calculate_indices=True,
    calculate_tocantins=True
)

# 2. Train model (automatically uses available years)
trainer = training.ConvLSTMTrainer(
    data_dir="data/processed",
    model_config="configs/model_config.yaml"
)
trainer.train(epochs=100, batch_size=8)

# 3. Predict any future year (must be beyond your data range)
predictor = prediction.FuturePredictor(
    model_path="models/convlstm_best.pth",
    input_year=2022  # Your most recent data year
)
predictions_2030 = predictor.predict(target_year=2030)

# 4. Analyze interventions
analyzer = analysis.InterventionAnalyzer(
    current_raster="data/processed/2022.tif",
    predicted_raster=predictions_2030
)
priorities = analyzer.identify_priority_zones(
    threshold="high",
    save_map=True
)
```

---

## Methodology

### Data Pipeline

```
Raw Landsat GeoTIFFs (YOUR CUSTOM DATE RANGE)
    ↓
Temporal Organization (configurable interval: annual, biennial, etc.)
    ↓
Feature Calculation per pixel:
  - NDBI (Normalized Difference Built-up Index)
  - NDVI (Normalized Difference Vegetation Index)
  - NDWI (Normalized Difference Water Index)
  - NDBSI (Normalized Difference Bareness and Soil Index)
  - LST (Land Surface Temperature)
  - Impact Score (IS) via Tocantins Framework
  - Severity Score (SS) via Tocantins Framework
    ↓
ConvLSTM Training (spatiotemporal sequence learning)
    ↓
Future Prediction (ANY YEAR beyond your data range)
    ↓
Residual Analysis (predicted - current)
    ↓
Intervention Priority Mapping
```

### Model Architecture

```python
ConvLSTM Encoder-Decoder:
  Input: (batch, time_steps, channels, height, width)
  - time_steps: Configurable based on your data (e.g., 10-20 timesteps)
  - channels: 7 (NDBI, NDVI, NDWI, NDBSI, LST, IS, SS)
  
  Encoder:
    - ConvLSTM Layer 1: 64 filters
    - ConvLSTM Layer 2: 128 filters
    - ConvLSTM Layer 3: 256 filters
  
  Decoder:
    - ConvLSTM Layer 4: 256 filters
    - ConvLSTM Layer 5: 128 filters
    - ConvLSTM Layer 6: 64 filters
    - Conv2D Output: 7 channels (predicted metrics)
```

---

## Configuration

### Preprocessing Config (`configs/preprocessing_config.yaml`)

```yaml
temporal:
  start_year: 2000  # YOUR earliest data year
  end_year: 2022    # YOUR most recent data year
  interval: 2       # 1 for annual, 2 for biennial, etc.
  season: "07-01_12-31"  # Seasonal composite (adjust as needed)

indices:
  calculate_ndbi: true
  calculate_ndvi: true
  calculate_ndwi: true
  calculate_ndbsi: true
  calculate_lst: true

tocantins:
  enabled: true
  k_threshold: 1.5
  spatial_params:
    min_anomaly_size: 1
    agglutination_distance: 4
```

### Model Config (`configs/model_config.yaml`)

```yaml
architecture: "convlstm"
input_channels: 7
hidden_dims: [64, 128, 256, 256, 128, 64]
kernel_size: 3
num_layers: 6
batch_first: true
bias: true
return_all_layers: false

training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.001
  optimizer: "adam"
  loss_function: "mse"
  
  # Adjust sequence_length based on your temporal resolution
  # For biennial data with 40 years (20 timesteps): sequence_length: 10
  # For annual data with 20 years: sequence_length: 10-15
  sequence_length: 10
  
  early_stopping:
    patience: 15
    min_delta: 0.0001
```

### Customizing Intervention Priority Weights

The intervention analysis uses a weighted combination of metrics to identify priority zones for urban heat mitigation. You can customize these weights in your configuration file or when using the `ResidualCalculator` directly.

#### Configuration File

```yaml
analysis:
  threshold: high  # low, medium, high, or custom value (0.0-1.0)
  
  # Priority scoring weights (configurable)
  priority_weights:
    LST: 0.4   # Land Surface Temperature - direct heat measure
    IS: 0.3    # Impact Score - spatial extent of thermal anomaly
    SS: 0.2    # Severity Score - intensity of thermal anomaly  
    NDBI: 0.1  # Normalized Difference Built-up Index
```

**Default weights** (if not specified): `LST=0.4, IS=0.3, SS=0.2, NDBI=0.1`

**Important**: Weights don't need to sum to 1.0 - they will be normalized automatically during the residual calculation.

#### Programmatic Usage

```python
from urbanai.analysis import ResidualCalculator

# Example: Emphasize LST over other metrics
analyzer = ResidualCalculator(
    current_raster="data/processed/2022_features_complete.tif",
    future_raster="predictions/2030_prediction.tif",
    weights={
        "LST": 0.5,   # Increase LST importance
        "IS": 0.25,   # Impact Score
        "SS": 0.20,   # Severity Score
        "NDBI": 0.05  # Built-up index
    }
)
residuals = analyzer.calculate_all_residuals()
```

**Metric Descriptions:**
- **LST** (Land Surface Temperature): Direct measure of surface heat
- **IS** (Impact Score): Spatial extent of thermal anomalies (from Tocantins Framework)
- **SS** (Severity Score): Intensity of thermal anomalies (from Tocantins Framework)
- **NDBI** (Normalized Difference Built-up Index): Proxy for urbanization level

---

## Input Data Requirements

### Landsat Collection

- **Satellite**: Landsat 5/7/8/9
- **Collection**: Level-2, Collection 2
- **Bands**: SR_B1-B7, ST_B10, QA_PIXEL
- **Temporal Range**: YOUR CUSTOM RANGE (e.g., 2000-2022, 1985-2023, etc.)
- **Temporal Resolution**: Annual, biennial, or custom intervals
- **Season**: Configurable (e.g., dry season, growing season)
- **Spatial Resolution**: 30m
- **Format**: GeoTIFF

### File Naming Convention

```
L[5|7|8|9]_GeoTIFF_YYYY-MM-DD_YYYY-MM-DD_cropped.tif

Examples:
L5_GeoTIFF_2000-07-01_2000-12-31_cropped.tif
L8_GeoTIFF_2022-07-01_2022-12-31_cropped.tif
```

### Important Notes on Temporal Configuration

1. **Start and End Years**: Define these based on YOUR available Landsat data
2. **Prediction Years**: Can be any year beyond your `end_year`
3. **Sequence Length**: Adjust based on your temporal interval:
   - Biennial data (40 years → 20 timesteps): use sequence_length 10-12
   - Annual data (20 years): use sequence_length 10-15
   - The model needs enough history to learn patterns
4. **Validation**: The framework automatically validates year consistency

### Google Earth Engine Script

Use the provided GEE script (included in repository) to export Landsat composites with cloud masking and scale factor application. The script is flexible and works for any date range.

---

## Output Files

### Processed Features

```
data/processed/
├── 2000_features.tif       # 7-band: NDBI, NDVI, NDWI, NDBSI, LST, IS, SS
├── 2002_features.tif
├── ...
└── 2022_features.tif
```

### Model Outputs

```
results/
├── models/
│   ├── convlstm_best.pth
│   ├── convlstm_last.pth
│   └── training_history.csv
├── predictions/
│   ├── 2030_predicted.tif      # Your target year
│   ├── 2030_residuals.tif      # change from latest data year
│   └── uncertainty_map.tif
└── analysis/
    ├── intervention_priorities.tif
    ├── intervention_priorities.geojson
    ├── hotspot_statistics.csv
    └── visualization/
        ├── temporal_evolution.png
        ├── prediction_map.png
        └── intervention_map.png
```

---

## Command Line Interface

```bash
# Preprocess data (automatically detects years from filenames)
urbanai preprocess --input data/raw --output data/processed --config configs/preprocessing_config.yaml

# Train model (uses all available processed years)
urbanai train --data data/processed --config configs/model_config.yaml --epochs 100

# Predict specific future year
urbanai predict --model models/convlstm_best.pth --year 2030 --output results/predictions

# Generate intervention map
urbanai analyze --current data/processed/2022_features.tif --predicted results/predictions/2030_predicted.tif --output results/analysis
```

---

## Validation and Error Handling

UrbanAI includes comprehensive validation:

```python
# The pipeline validates:
# 1. Target year is beyond your data range
if target_year <= current_year:
    raise ValueError(f"Target year must be > {current_year}")

# 2. Sufficient historical data for training
if n_years < sequence_length:
    raise ValueError(f"Need at least {sequence_length} years of data")

# 3. Consistent temporal intervals
# Automatically detects and validates your temporal spacing
```

---

## Best Practices

### Temporal Configuration

1. **Data Availability**: Use continuous temporal data without large gaps
2. **Minimum History**: At least 10 timesteps recommended for robust training
3. **Prediction Horizon**: Longer predictions (>10 years) increase uncertainty
4. **Validation Split**: Reserve latest years for validation (automatic in framework)

### Example Configurations

```python
# Good: 40 years of biennial data (20 timesteps)
config = {
    "preprocessing": {
        "start_year": 1985,
        "end_year": 2023,
        "interval": 2
    },
    "training": {
        "sequence_length": 10
    }
}

# Good: 15 years of annual data
config = {
    "preprocessing": {
        "start_year": 2008,
        "end_year": 2022,
        "interval": 1
    },
    "training": {
        "sequence_length": 12
    }
}

# Caution: Too few timesteps
config = {
    "preprocessing": {
        "start_year": 2015,
        "end_year": 2022,
        "interval": 2  # Only 4 timesteps!
    }
}
# This will raise a validation error
```

---

## Scientific Foundation

UrbanAI implements methodologies combining:

1. **ConvLSTM Architecture**: Shi et al. (2015) - "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
2. **Tocantins Framework**: Borges (2025) - For thermal anomaly detection and quantification
3. **Urban Heat Island Analysis**: Standard remote sensing indices and thermal analysis

### Citation

If you use UrbanAI in your research, please cite:

```bibtex
@software{urbanai_2025,
  author = {Borges, Isaque Carvalho},
  title = {UrbanAI: Deep Learning Framework for Spatiotemporal Urban Heat Prediction},
  year = {2025},
  publisher = {EcoAção Brasil},
  url = {https://github.com/EcoAcao-Brasil/ecoacao-urbanai}
}

@software{tocantins_framework_2025,
  author = {Borges, Isaque Carvalho},
  title = {Tocantins Framework: A Python Library for Assessment of Intra-Urban Thermal Anomaly},
  year = {2025},
  publisher = {EcoAção Brasil},
  url = {https://github.com/EcoAcao-Brasil/tocantins-framework}
}
```

---

## Contributing

We welcome contributions from the research community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- **Email**: isaque@ecoacaobrasil.org
- **Issues**: [GitHub Issues](https://github.com/EcoAcao-Brasil/ecoacao-urbanai/issues)

---

## Keywords

urban heat island, deep learning, ConvLSTM, spatiotemporal prediction, remote sensing, Landsat, climate adaptation, thermal anomaly detection, urban planning, machine learning, PyTorch, geospatial analysis, time series forecasting, modular framework
