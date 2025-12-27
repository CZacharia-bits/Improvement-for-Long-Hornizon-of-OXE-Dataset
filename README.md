# OXE Dataset Visualization

A Python toolkit for loading and visualizing robotics data from the [Open X-Embodiment](https://robotics-transformer-x.github.io/) dataset.

![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-%3E%3D2.13-orange.svg)

---

## Quick Start

```bash
make install        # Install dependencies
make visualize-all  # Generate visualizations for all 12 datasets
```

---

## Data Specification

### Storage Format

The OXE dataset is stored as **TFRecord** files in Google Cloud Storage:

```
gs://x-embodiment-imporvement/oxe_v1_0/{dataset_name}/{version}/{dataset_name}-{split}.tfrecord-*
```

### Data Structure

Each TFRecord contains **trajectories** (episodes) with the following hierarchy:

```
Trajectory
├── steps/
│   ├── observation/
│   │   ├── rgb_static        # Static camera RGB image (JPEG encoded)
│   │   ├── rgb_gripper       # Gripper camera RGB image
│   │   ├── image             # Primary image (dataset-dependent)
│   │   └── ...               # Additional sensor data
│   ├── action/               # Robot action commands
│   └── reward/               # Task reward signal
└── metadata/                 # Episode-level information
```

### Key Feature Types

| Feature Key | Type | Description |
|------------|------|-------------|
| `steps/observation/rgb_static` | `VarLenFeature(string)` | JPEG-encoded image sequence from static camera |
| `steps/observation/rgb_gripper` | `VarLenFeature(string)` | JPEG-encoded image sequence from gripper camera |
| `steps/observation/image` | `VarLenFeature(string)` | Primary observation image |
| `steps/action/*` | `VarLenFeature(float)` | Robot action parameters |

> **Note**: Available features vary by dataset. Use `make inspect` to explore a specific dataset's structure.

---

## Inputs / Outputs

### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | `str` | `"taco_play"` | Name of the OXE dataset |
| `split` | `str` | `"train"` | Dataset split (`train`, `val`) |
| `n` | `int` | `8` | Number of frames to visualize |
| `image_key` | `str` | Auto-detected | Feature key for images |

### Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Visualization | PNG | Grid of N frames from first trajectory |
| Inspection | Console | List of all feature keys in dataset |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   config.py          Defines GCS bucket & path structure    │
│       │                                                     │
│       ▼                                                     │
│   data_loader.py     Constructs GCS pattern, loads TFRecords│
│       │              using tf.data.TFRecordDataset          │
│       ▼                                                     │
│   visualize_images.py                                       │
│       │  1. Auto-detects image feature key                  │
│       │  2. Parses VarLenFeature to extract image bytes     │
│       │  3. Decodes JPEG → RGB tensor                       │
│       │  4. Renders grid with matplotlib                    │
│       ▼                                                     │
│   {dataset}_first_frames.png                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key algorithms:**
- **Auto-detection**: Scans feature keys for `rgb_static` → `image` → `rgb_gripper` → any key containing "image/rgb"
- **Parallel loading**: Uses `tf.data.AUTOTUNE` for optimized I/O
- **Variable-length parsing**: Handles trajectories of different lengths via `tf.io.VarLenFeature`

---

## Components

| Module | Purpose |
|--------|---------|
| `config.py` | GCS bucket configuration (`OXEDatasetConfig` dataclass) |
| `data_loader.py` | `gcs_pattern()` and `load_raw()` for TFRecord loading |
| `visualize_images.py` | `visualize_first_n()` and `visualize_all_datasets()` |
| `inspect_example.py` | `inspect_one()` to print dataset feature keys |
| `list_datasets.py` | `list_available_datasets()` from GCS bucket |

---

## Available Datasets

| # | Dataset | Version | Robot Type |
|---|---------|---------|------------|
| 1 | `berkeley_autolab_ur5` | 0.1.0 | UR5 |
| 2 | `berkeley_cable_routing` | 0.1.0 | Franka |
| 3 | `bridge` | 0.1.0 | WidowX |
| 4 | `fractal20220817_data` | 0.1.0 | Franka |
| 5 | `jaco_play` | 0.1.0 | Jaco |
| 6 | `kuka` | 0.1.0 | Kuka IIWA |
| 7 | `language_table` | 0.0.1 | xArm |
| 8 | `nyu_door_opening_surprising_effectiveness` | 0.1.0 | Hello Robot |
| 9 | `roboturk` | 0.1.0 | Sawyer |
| 10 | `taco_play` | 0.1.0 | Franka |
| 11 | `toto` | 0.1.0 | Franka |
| 12 | `viola` | 0.1.0 | Franka |

---

## Usage

```bash
make help           # Show all commands
make visualize      # Visualize taco_play only
make visualize-all  # Visualize all datasets
make inspect        # Show feature keys for taco_play
make list-datasets  # List available datasets
make clean          # Remove generated PNGs
```

**Python API:**

```python
from src.oxe_viz.visualize_images import visualize_first_n
from src.oxe_viz.inspect_example import inspect_one

visualize_first_n("bridge", split="train", n=10, out_path="bridge_viz.png")
inspect_one("kuka", "train")
```

---

## Project To-Do

- [ ] Add action trajectory visualization (plot joint positions over time)
- [ ] Add video export (MP4/GIF from image sequences)
- [ ] Support reward/success visualization
- [ ] Multi-trajectory comparison view
- [ ] Interactive web-based viewer
- [ ] Add unit tests and CI pipeline
- [ ] Publish as pip-installable package

---

## Requirements

- Python 3.x
- TensorFlow ≥ 2.13
- Matplotlib
- GCS access (datasets are public)

---

## License

MIT
