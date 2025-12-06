# OXE Dataset Visualization

A Python package for loading and visualizing data from the Open X-Embodiment (OXE) dataset stored in Google Cloud Storage as TFRecord files.

## Overview

This package provides utilities to:
- Load TFRecord datasets from Google Cloud Storage
- Visualize image sequences from the dataset
- Inspect dataset structure and feature keys

## Project Structure

```
oxe-viz/
├── src/
│   └── oxe_viz/
│       ├── __init__.py
│       ├── config.py          # Configuration for GCS paths
│       ├── data_loader.py      # Functions to load TFRecord datasets
│       ├── visualize_images.py # Visualization utilities
│       └── inspect_example.py  # Dataset inspection utilities
├── requirements.txt
├── Makefile
└── README.md
```

## Installation

Install the required dependencies:

```bash
make install
```

Or manually:

```bash
pip install -r requirements.txt
```

## Usage

### Visualize Dataset Images

Generate a visualization of the first N frames from a dataset:

```bash
make visualize
```

This will:
- Load the `taco_play` dataset from the train split
- Extract the first 8 frames from the first trajectory
- Save the visualization as `taco_play_first_frames.png`

### Inspect Dataset Structure

Inspect the structure of a dataset to see available feature keys:

```bash
make inspect
```

This will print all feature keys in the dataset and highlight image-related keys.

### Custom Usage

You can also use the functions programmatically:

```python
from src.oxe_viz.visualize_images import visualize_first_n
from src.oxe_viz.inspect_example import inspect_one
from src.oxe_viz.data_loader import load_raw

# Visualize with custom parameters
visualize_first_n(
    dataset_name="taco_play",
    split="train",
    n=10,
    out_path="custom_output.png"
)

# Inspect a different dataset
inspect_one("language_table", "train")

# Load raw dataset
dataset = load_raw("taco_play", "train")
```

## How It Works

### Architecture

1. **config.py**: Defines the `OXEDatasetConfig` class that specifies the GCS bucket and path structure for OXE datasets.

2. **data_loader.py**: 
   - `gcs_pattern()`: Constructs the GCS path pattern for a given dataset and split
   - `load_raw()`: Loads TFRecord files from GCS and returns a TensorFlow dataset

3. **visualize_images.py**:
   - `parse_example()`: Parses a TFRecord example to extract image sequences
   - `visualize_first_n()`: Loads a dataset, extracts the first N frames, and saves them as a visualization

4. **inspect_example.py**:
   - `inspect_one()`: Loads a single example and prints all available feature keys

### Data Flow

```
config.py (GCS paths)
    ↓
data_loader.py (load TFRecord files)
    ↓
visualize_images.py / inspect_example.py (process and visualize)
```

### Supported Datasets

The package supports datasets from the OXE v1.0 collection. There are **12 available datasets**:

1. `berkeley_autolab_ur5`
2. `berkeley_cable_routing`
3. `bridge`
4. `fractal20220817_data`
5. `jaco_play`
6. `kuka`
7. `language_table` (uses version `0.0.1`)
8. `nyu_door_opening_surprising_effectiveness`
9. `roboturk`
10. `taco_play` (default)
11. `toto`
12. `viola`

**List all available datasets:**
```bash
make list-datasets
```

By default, the package uses:
- Dataset: `taco_play`
- Split: `train`
- Version: `0.1.0` (or `0.0.1` for `language_table`)

The dataset is loaded from: `gs://x-embodiment-imporvement/oxe_v1_0/{dataset_name}/{version}/{dataset_name}-{split}.tfrecord-*`

## Requirements

- Python 3.x
- TensorFlow >= 2.13
- Matplotlib

## Output

The visualization script generates PNG images showing a grid of frames from the dataset. The default output is `taco_play_first_frames.png`.

## Cleanup

Remove generated image files:

```bash
make clean
```

## Notes

- The visualization uses the static camera view (`steps/observation/rgb_static`) by default. You can modify `IMAGE_FEATURE_KEY` in `visualize_images.py` to use the gripper view (`steps/observation/rgb_gripper`) instead.
- The package requires access to Google Cloud Storage to load datasets.
- Image sequences are stored as variable-length features in the TFRecord format.

