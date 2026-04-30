# UrbanAI

**ConvLSTM-based prediction of urban heat from multitemporal Landsat imagery**

Developed by [EcoAção Brasil](https://ecoacaobrasil.org)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

UrbanAI trains a ConvLSTM encoder-decoder on a temporal sequence of Landsat-derived rasters and outputs a predicted raster for a user-specified future year. The predicted raster contains the same spectral indices as the training data (NDBI, NDVI, NDWI, NDBSI, LST and, optionally, Impact Score and Severity Score from the Tocantins Framework).

**Current scope:** preprocessing → training → predicted raster output.  
Comparison of predictions against a future observed state and intervention guidance are planned for a later version.

---

## Installation

```bash
pip install urbanai
```

**Requirements:** Python 3.8+, GDAL (install via conda: `conda install -c conda-forge gdal`)

---

## Quick Start

```python
from urbanai import UrbanAIPipeline

pipeline = UrbanAIPipeline(
    input_dir="data/raw",      # directory of raw Landsat GeoTIFFs
    output_dir="results",
    config={
        "preprocessing": {
            "start_year": 2000,
            "end_year": 2022,
            "interval": 2,       # biennial
        },
        "training": {
            "epochs": 100,
            "batch_size": 8,
        },
    },
)

results = pipeline.run(predict_year=2030)
# Output: results/predictions/2030_predicted.tif
```

---

## Input Data

- **Satellite:** Landsat 5 / 7 / 8 / 9, Level-2 Collection 2
- **Required bands:** SR_B1–B7, ST_B10
- **Format:** GeoTIFF, one composite per year
- **Naming convention:** `{YEAR}_{description}_cropped.tif`

---

## Pipeline

```
Raw Landsat GeoTIFFs
    ↓ TemporalDataProcessor
Spectral index rasters: NDBI, NDVI, NDWI, NDBSI, LST
    [+ IS, SS if Tocantins is enabled]
    ↓ UrbanAITrainer
ConvLSTM checkpoint (convlstm_best.pth)
    ↓ FuturePredictor
{YEAR}_predicted.tif
```

### Tocantins Framework (optional)

When `preprocessing.tocantins.enabled: true`, two additional channels are calculated using the [Tocantins Framework](https://github.com/EcoAcao-Brasil/tocantins-framework):
- **IS** (Impact Score) — spatial extent of the thermal anomaly
- **SS** (Severity Score) — intensity of the thermal anomaly

This increases the model channel count from 5 to 7. Set `enabled: false` to skip this step and use only the five spectral indices.

---

## Configuration

Copy `configs/complete_config.yaml` and adjust for your data. Key parameters:

```yaml
preprocessing:
  start_year: 2000    # earliest available data year
  end_year: 2022      # most recent available data year
  interval: 2         # temporal interval in years
  landsat_version: 8
  tocantins:
    enabled: false    # set true to include IS/SS

training:
  epochs: 100
  batch_size: 8
  sequence_length: 10   # input sequence length
  normalization_method: zscore
  gradient_accumulation_steps: 1  # increase to reduce memory usage

prediction:
  calculate_uncertainty: false  # enable MC dropout uncertainty
```

### Gradient accumulation

For limited RAM, set `gradient_accumulation_steps > 1`:

| RAM  | batch_size | gradient_accumulation_steps | Effective batch |
|------|-----------|----------------------------|-----------------|
| 24GB | 8         | 1                          | 8               |
| 12GB | 2         | 4                          | 8               |
| 8GB  | 1         | 8                          | 8               |

---

## CLI

```bash
# Run complete pipeline
urbanai run --input data/raw --output results --predict-year 2030

# Run individual stages
urbanai preprocess --input data/raw --output results/processed
urbanai train --data results/processed --output results/models
urbanai predict --model results/models/convlstm_best.pth \
                --data results/processed --year 2030 --output results/predictions
```

---

## Module API

Each stage can be used independently:

```python
from urbanai.preprocessing import TemporalDataProcessor
from urbanai.training import UrbanAITrainer
from urbanai.prediction import FuturePredictor

# Preprocess
processor = TemporalDataProcessor(raw_dir="data/raw", output_dir="data/processed")
processor.process_all_years()

# Train
trainer = UrbanAITrainer(data_dir="data/processed", output_dir="models")
trainer.train(epochs=100, batch_size=8)

# Predict
predictor = FuturePredictor(
    model_path="models/convlstm_best.pth",
    data_dir="data/processed",
    output_dir="predictions",
)
predictor.predict(current_year=2022, target_year=2030)
```

---

## Model Architecture

ConvLSTM encoder-decoder operating on sequences of shape `(batch, time, channels, H, W)`.

```
Encoder:
  ConvLSTM  64 filters
  ConvLSTM 128 filters
  ConvLSTM 256 filters

Decoder:
  ConvLSTM 256 filters
  ConvLSTM 128 filters
  ConvLSTM  64 filters
  Conv2D → 5 or 7 output channels
```

---

## Output

`{YEAR}_predicted.tif` — a multi-band GeoTIFF with the same CRS and spatial extent as the input data, one band per predicted index.

Optional: `{YEAR}_uncertainty.tif` when `calculate_uncertainty: true` (Monte Carlo dropout).

---

## License

MIT — see [LICENSE](LICENSE)

Contact: isaque@ecoacaobrasil.org

