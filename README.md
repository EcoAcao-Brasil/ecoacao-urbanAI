# UrbanAI, by EcoAção Brasil (Brazil EcoAction).

**Deep learning framework for spatiotemporal urban heat prediction using ConvLSTM and satellite imagery**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/urbanai.svg)](https://badge.fury.io/py/urbanai)

Developed by [EcoAção Brasil](https://ecoacaobrasil.org)

---

## Overview

UrbanAI is a comprehensive deep learning framework for assessing the evolution of the Urban Heat Island (UHI) phenomenon with an emphasis on urban climate planning. Built on ConvLSTM architecture, it analyzes historical Landsat data (1985-2025) to forecast future thermal landscapes and identify priority areas for urban climate intervention.

### Key Features

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

### Basic Usage

```python
from urbanai import UrbanAIPipeline

# Initialize pipeline
pipeline = UrbanAIPipeline(
    input_dir="data/raw",
    output_dir="results",
    config="configs/default_config.yaml"
)

# Run complete workflow
pipeline.run(
    preprocess=True,
    train=True,
    predict_year=2030,
    analyze_interventions=True
)

# Access results
predictions = pipeline.get_predictions()
intervention_map = pipeline.get_intervention_priorities()
```

### Step-by-Step Workflow

```python
from urbanai import preprocessing, models, training, prediction, analysis

# 1. Prepare data
processor = preprocessing.TemporalDataProcessor(
    raw_dir="data/raw",
    output_dir="data/processed"
)
processor.process_all_years(
    years=[1985, 1987, ..., 2023, 2025],
    calculate_indices=True,
    calculate_tocantins=True
)

# 2. Train model
trainer = training.ConvLSTMTrainer(
    data_dir="data/processed",
    model_config="configs/model_config.yaml"
)
trainer.train(epochs=100, batch_size=8)

# 3. Predict future
predictor = prediction.FuturePredictor(
    model_path="models/convlstm_best.pth",
    input_year=2025
)
predictions_2030 = predictor.predict(target_year=2030)

# 4. Analyze interventions
analyzer = analysis.InterventionAnalyzer(
    current_raster="data/processed/2025.tif",
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
Raw Landsat GeoTIFFs (1985-2025)
    ↓
Temporal Organization (biennial: 1985, 1987, ..., 2025)
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
Future Prediction (2030/2035)
    ↓
Residual Analysis (2030 - 2025)
    ↓
Intervention Priority Mapping
```

### Model Architecture

```python
ConvLSTM Encoder-Decoder:
  Input: (batch, time_steps, channels, height, width)
  - time_steps: 20 (1985-2025, biennial)
  - channels: 7 (NDBI, NDVI, NDWI, NDBSI, LST, IS, SS)
  
  Encoder:
    - ConvLSTM Layer 1: 64 filters
    - ConvLSTM Layer 2: 128 filters
    - ConvLSTM Layer 3: 256 filters
  
  Decoder:
    - ConvLSTM Layer 4: 256 filters
    - ConvLSTM Layer 5: 128 filters
    - ConvLSTM Layer 6: 64 filters
    - Conv2D Output: 7 channels (predicted metrics for 2030)
```

---

## Configuration

### Preprocessing Config (`configs/preprocessing_config.yaml`)

```yaml
temporal:
  start_year: 1985
  end_year: 2025
  interval: 2  # biennial
  season: "07-01_12-31"  # July-December composite

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
  early_stopping:
    patience: 15
    min_delta: 0.0001
```

---

## Input Data Requirements

### Landsat Collection

- **Satellite**: Landsat 5/7/8/9
- **Collection**: Level-2, Collection 2
- **Bands**: SR_B1-B7, ST_B10, QA_PIXEL
- **Temporal Range**: 1985-2025 (biennial composites)
- **Season**: July-December (dry season in Palmas, Tocantins)
- **Spatial Resolution**: 30m
- **Format**: GeoTIFF

### File Naming Convention

```
L[5|7|8]_GeoTIFF_YYYY-07-01_YYYY-12-31_cropped.tif

Examples:
L5_GeoTIFF_1985-07-01_1985-12-31_cropped.tif
L8_GeoTIFF_2023-07-01_2023-12-31_cropped.tif
```

### Google Earth Engine Script

Use the provided GEE script (included in repository) to export Landsat composites with cloud masking and scale factor application.

---

## Output Files

### Processed Features

```
data/processed/
├── 1985_features.tif       # 7-band: NDBI, NDVI, NDWI, NDBSI, LST, IS, SS
├── 1987_features.tif
├── ...
└── 2025_features.tif
```

### Model Outputs

```
results/
├── models/
│   ├── convlstm_best.pth
│   ├── convlstm_last.pth
│   └── training_history.csv
├── predictions/
│   ├── 2030_predicted.tif
│   ├── 2030_residuals.tif  # 2030 - 2025
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

## API Reference

### Core Modules

#### `urbanai.preprocessing`
- `RasterLoader`: Load and validate GeoTIFF files
- `BandProcessor`: Calculate spectral indices and LST
- `TocantinsIntegration`: Compute IS and SS metrics
- `TemporalDataProcessor`: Orchestrate temporal data preparation

#### `urbanai.models`
- `ConvLSTM`: ConvLSTM cell implementation
- `EncoderDecoder`: Encoder-decoder architecture
- `ModelFactory`: Model instantiation with configs

#### `urbanai.training`
- `UrbanAITrainer`: Training orchestration
- `UrbanDataset`: PyTorch Dataset for spatiotemporal data
- `SpatialLoss`: Custom loss functions for spatial data
- `TrainingCallbacks`: Early stopping, checkpointing, logging

#### `urbanai.prediction`
- `FuturePredictor`: Inference engine for future forecasting
- `EnsemblePredictor`: Multi-model ensemble predictions
- `UncertaintyEstimator`: Prediction uncertainty quantification

#### `urbanai.analysis`
- `ResidualCalculator`: Compute temporal changes
- `InterventionAnalyzer`: Identify priority intervention zones
- `TrendAnalyzer`: Temporal trend analysis

---

## Examples

### Jupyter Notebooks

See `examples/notebooks/` for comprehensive tutorials:

1. **Data Preparation** (`01_data_preparation.ipynb`)
2. **Model Training** (`02_model_training.ipynb`)
3. **Future Prediction** (`03_prediction_2030.ipynb`)
4. **Intervention Mapping** (`04_intervention_mapping.ipynb`)

### Command Line Interface

```bash
# Preprocess data
urbanai preprocess --input data/raw --output data/processed --config configs/preprocessing_config.yaml

# Train model
urbanai train --data data/processed --config configs/model_config.yaml --epochs 100

# Predict future
urbanai predict --model models/convlstm_best.pth --year 2030 --output results/predictions

# Generate intervention map
urbanai analyze --current data/processed/2025_features.tif --predicted results/predictions/2030_predicted.tif --output results/analysis
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

### Development Setup

```bash
git clone https://github.com/EcoAcao-Brasil/ecoacao-urbanai
cd ecoacao-urbanai
pip install -e ".[dev]"
pre-commit install
pytest tests/
```

### Code Standards

- **Style**: Black, isort
- **Linting**: Ruff, mypy
- **Testing**: pytest, >80% coverage
- **Documentation**: Google-style docstrings

---

## Roadmap

- [x] Core ConvLSTM implementation
- [x] Tocantins Framework integration
- [ ] Multi-GPU training support
- [ ] Transformer-based alternatives (ConvTransformer)
- [ ] EcoAction Platform API integration
- [ ] Real-time prediction service
- [ ] Multi-city benchmarking dataset
- [ ] Transfer learning for new cities

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Developed by **EcoAção Brasil** to support climate resilience research and urban planning initiatives.

Special thanks to the open-source community and contributors of:
- PyTorch
- Rasterio
- GDAL
- Tocantins Framework

---

## Support

- **Email**: isaque@ecoacaobrasil.org
- **Issues**: [GitHub Issues](https://github.com/EcoAcao-Brasil/ecoacao-urbanai/issues)
- **Documentation**: [Read the Docs](https://ecoacao-urbanai.readthedocs.io)

---

## Keywords

urban heat island, deep learning, ConvLSTM, spatiotemporal prediction, remote sensing, Landsat, climate adaptation, thermal anomaly detection, urban planning, machine learning, PyTorch, geospatial analysis
