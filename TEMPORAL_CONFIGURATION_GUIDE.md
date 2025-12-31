# UrbanAI's Supercalifragilisticexpialidocious Temporal Configuration Guide, by EcoAção Brasil (Brazil EcoAction).

This document provides technical specifications for configuring temporal parameters in UrbanAI. The framework does not make assumptions about data year ranges and requires explicit user configuration.

## Overview

UrbanAI requires configuration of the following temporal parameters:
- Start year: earliest available Landsat data
- End year: most recent available Landsat data
- Prediction target year: future year for forecasting
- Temporal interval: annual (1), biennial (2), or custom

## Configuration Syntax

```yaml
# Configuration file format:
preprocessing:
  start_year: YYYY    # Example: 2000, 1985, 2010
  end_year: YYYY      # Example: 2022, 2023, 2020
  interval: N         # 1 (annual), 2 (biennial)

# Pipeline execution:
pipeline.run(predict_year=YYYY)  # Constraint: predict_year > end_year
```

## Determining Available Data Range

Programmatic detection of available Landsat data years:

```python
from pathlib import Path
import re

data_dir = Path("data/raw")
files = sorted(data_dir.glob("*_cropped.tif"))

years = []
for f in files:
    match = re.search(r"(\d{4})", f.name)
    if match:
        years.append(int(match.group(1)))

available_years = sorted(set(years))
start_year = min(years)
end_year = max(years)

print(f"Available years: {available_years}")
print(f"Data range: {start_year} to {end_year}")
```

## Preprocessing Configuration

Configuration based on detected data range:

```yaml
# configs/preprocessing_config.yaml
preprocessing:
  start_year: 2000  # Detected minimum year
  end_year: 2022    # Detected maximum year
  interval: 2       # 1 for annual, 2 for biennial
```

## Training Parameter Configuration

Sequence length calculation methodology:

```yaml
training:
  # Calculation formula: sequence_length = n_timesteps * 0.6 to 0.7
  # 
  # Examples:
  # 40-year biennial (20 timesteps): sequence_length: 10-12
  # 20-year annual (20 timesteps): sequence_length: 10-12
  # 10-year annual (10 timesteps): sequence_length: 6-8
  
  sequence_length: 10
  prediction_horizon: 1
```

## Prediction Target Selection

Target year must exceed the end year defined in preprocessing configuration:

```python
pipeline.run(
    predict_year=2030  # Constraint: 2030 > end_year
)

# Valid examples by end_year:
# end_year = 2022: valid targets are 2023, 2025, 2030, 2035, etc.
# end_year = 2020: valid targets are 2021, 2025, 2030, 2035, etc.
# end_year = 2023: valid targets are 2024, 2028, 2030, 2035, etc.
```

## Configuration Examples

### Example 1: Long-term Historical Analysis

Data specification: Landsat 1985-2023 (biennial intervals)

```yaml
preprocessing:
  start_year: 1985
  end_year: 2023
  interval: 2

training:
  sequence_length: 10  # 19 timesteps available, use 10
  epochs: 100
  batch_size: 8
```

```python
# 7-year forecast
pipeline.run(predict_year=2030)

# 12-year forecast
pipeline.run(predict_year=2035)
```

Technical rationale:
- Total timesteps: 20 (1985, 1987, ..., 2023)
- Sequence length: 10 (50% of available timesteps)
- Forecast horizon: 7-12 years considered reasonable for this temporal extent

---

### Example 2: Recent High-Resolution Analysis

Data specification: Landsat 2015-2023 (annual intervals)

```yaml
preprocessing:
  start_year: 2015
  end_year: 2023
  interval: 1

training:
  sequence_length: 6  # 9 timesteps available, use 6
  epochs: 100
  batch_size: 8
```

```python
# 5-year forecast
pipeline.run(predict_year=2028)
```

Technical rationale:
- Total timesteps: 9 (2015-2023, annual)
- Sequence length: 6 (67% of available timesteps)
- Shorter forecast horizon (5 years) due to limited training data

---

### Example 3: Mid-term Regional Analysis

Data specification: Landsat 2000-2020 (biennial intervals)

```yaml
preprocessing:
  start_year: 2000
  end_year: 2020
  interval: 2

training:
  sequence_length: 8  # 11 timesteps available, use 8
  epochs: 100
  batch_size: 8
```

```python
# 5-year forecast
pipeline.run(predict_year=2025)

# 10-year forecast
pipeline.run(predict_year=2030)
```

Technical rationale:
- Total timesteps: 11 (2000, 2002, ..., 2020)
- Sequence length: 8 (73% of available timesteps)
- Medium-term forecasts: 5-10 years

---

### Example 4: High-Density Recent Coverage

Data specification: Landsat 2010-2023 (annual intervals)

```yaml
preprocessing:
  start_year: 2010
  end_year: 2023
  interval: 1

training:
  sequence_length: 10  # 14 timesteps available, use 10
  epochs: 100
  batch_size: 8
```

```python
# 7-year forecast
pipeline.run(predict_year=2030)
```

Technical rationale:
- Total timesteps: 14 (2010-2023, annual)
- High temporal resolution
- Sufficient training history for robust forecasting

## Validation Requirements

UrbanAI implements automatic validation of temporal configuration.

### Valid Configurations

```python
# Configuration: end_year = 2022
pipeline.run(predict_year=2025)  # Valid: 2025 > 2022
pipeline.run(predict_year=2030)  # Valid: 2030 > 2022
pipeline.run(predict_year=2050)  # Valid: 2050 > 2022

# Configuration: end_year = 2020
pipeline.run(predict_year=2025)  # Valid: 2025 > 2020
```

### Invalid Configurations

```python
# Configuration: end_year = 2022
pipeline.run(predict_year=2022)  # Invalid: target <= current
pipeline.run(predict_year=2020)  # Invalid: target < current
pipeline.run(predict_year=2015)  # Invalid: target < start_year

# Error output:
# ValueError: Target year (2022) must be greater than the most recent data year (2022)
```

### Sequence Length Validation

```python
# Configuration: 5 timesteps available (2018-2022, annual)
# start_year: 2018, end_year: 2022, interval: 1

# Invalid configuration:
training: 
  sequence_length: 10  # Error: insufficient timesteps

# Valid configuration:
training:
  sequence_length: 3  # Valid: 3 < 5 available timesteps
```

## Technical Specifications

### Minimum Data Requirements

- Absolute minimum: 8-10 timesteps
- Recommended: 12-20 timesteps
- Optimal: 20+ timesteps for long-term forecasting

### Sequence Length Guidelines

Calculation formula:
```
sequence_length = available_timesteps * (0.6 to 0.7)
```

Application examples:
- 20 timesteps: sequence_length range 10-12
- 15 timesteps: sequence_length range 9-10
- 10 timesteps: sequence_length range 6-7

### Forecast Horizon Specifications

Classification by uncertainty level:
- Conservative: 5-7 years
- Moderate: 7-10 years
- Extended: 10-15 years
- High uncertainty: >15 years

Uncertainty increases with:
- Longer forecast horizons
- Fewer training timesteps
- Irregular temporal intervals

### Data Quality Considerations

Preferred configuration:
```yaml
# 10 years of consistent annual data
start_year: 2013
end_year: 2022
interval: 1
```

Problematic configuration:
```yaml
# 30 years with gaps and irregular intervals
# Available years: 1990, 1995, 2000, 2005, 2015, 2020
# Missing: 2010, irregular spacing
```

## Automatic Year Detection

UrbanAI implements automatic detection of available data years:

```python
from urbanai import UrbanAIPipeline

pipeline = UrbanAIPipeline(
    input_dir="data/raw",
    output_dir="results",
    config="config.yaml"
)

# Automatic detection process:
# 1. Scans processed data directory
# 2. Extracts years from filenames
# 3. Identifies most recent year
# 4. Uses detected year as prediction baseline

pipeline.run(predict_year=2030)
# Internal process: current_year = detected_year (e.g., 2022)
# Forecast calculation: 2030 - 2022 = 8-year horizon
```

## Troubleshooting

### Issue: No Processed Files Detected

Diagnostic procedure:

```python
from pathlib import Path
import re

raw_files = list(Path("data/raw").glob("*.tif"))
print([f.name for f in raw_files])

years = [int(re.search(r"(\d{4})", f.name).group(1)) for f in raw_files]
print(f"Detected years: {sorted(years)}")
```

Resolution: Update configuration to match detected years.

### Issue: Invalid Target Year

Error: "Target year must be greater than current year"

Resolution: Increase predict_year parameter:

```python
# If current_year = 2022
pipeline.run(predict_year=2025)  # Valid: 2025 > 2022
```

### Issue: Insufficient Training Data

Error: "Need at least X years of data"

Resolution options:
1. Acquire additional historical data
2. Reduce sequence_length parameter

```yaml
training:
  sequence_length: 6  # Reduced from 10
```

### Issue: Raster Dimension Mismatch

Error: "Shape mismatch in raster files"

Resolution requirements:
- Verify consistent spatial coverage in GEE export
- Validate identical dimensions across all files
- Re-export data if necessary

## Complete Configuration Example

Configuration file specification:

```yaml
# config_2010_2022.yaml
preprocessing:
  start_year: 2010
  end_year: 2022
  interval: 2
  landsat_version: 8

model:
  input_channels: 7
  hidden_dims: [64, 128, 256, 256, 128, 64]
  kernel_size: 3

training:
  epochs: 100
  batch_size: 8
  sequence_length: 5  # 7 timesteps (2010-2022), use 5
  learning_rate: 0.001
```

Pipeline execution:

```python
from urbanai import UrbanAIPipeline

pipeline = UrbanAIPipeline(
    input_dir="data/landsat_2010_2022",
    output_dir="results/analysis_2030",
    config="config_2010_2022.yaml"
)

results = pipeline.run(
    preprocess=True,
    train=True,
    predict_year=2030,
    analyze_interventions=True
)

print(f"Status: {results['status']}")
print(f"Output: {results['predictions']['output_path']}")
```

## Configuration Guidelines Summary

Required practices:
- Define start_year and end_year based on available data
- Select predict_year beyond data range
- Adjust sequence_length based on available timesteps
- Validate data coverage before training execution

Bad practices:
- Using framework default values without validation
- Attempting predictions within training range
- Configuring sequence_length exceeding available timesteps
- Combining irregular temporal intervals without adjustment

---

Technical support: isaque@ecoacaobrasil.org
