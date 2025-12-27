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

## What Is This Dataset For?

The OXE dataset contains **expert demonstrations** — videos of robots successfully performing tasks like picking, placing, and manipulating objects. All trajectories are successful (no failure cases).

### The Goal: Train One Model for Many Robots

Instead of training separate AI models for each robot type, researchers use OXE to train **generalist policies** — a single neural network that works across different robot arms (UR5, Franka, KUKA, etc.).

### What Are We Predicting?

**Frame-by-frame action prediction (Behavior Cloning):**

```
[Current camera image] + ["Pick up the blue cube"] → Model → [Next action]
```

The model predicts **7 numbers** per frame:

| Output | Description |
|--------|-------------|
| Δx, Δy, Δz | How much to move the gripper (position) |
| Δroll, Δpitch, Δyaw | How much to rotate the gripper (orientation) |
| gripper | Open (1) or close (0) |

### Why Only 7 Numbers for Complex Robot Arms?

Different robots have different numbers of joints (6-DoF, 7-DoF, etc.), but the AI doesn't predict joint angles. Instead:

1. **AI predicts:** Where the gripper should go (end-effector position)
2. **Robot calculates:** What joint angles achieve that position (Inverse Kinematics)

This abstraction lets the same model work on any robot — the robot handles its own body mechanics.

### Cross-Embodiment Challenges

| Challenge | How It's Handled |
|-----------|------------------|
| **Different control frequencies** (3-10 Hz) | Not standardized — models learn to handle variation |
| **Different units** (m, cm, radians) | Normalized during preprocessing |
| **Different joint counts** | Abstracted via end-effector predictions |
| **Different grippers** | Simplified to 1D open/close signal |
| **Different camera positions** | Models learn visual invariance from diverse data |

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
