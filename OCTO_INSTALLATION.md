# Octo Installation Guide

## Correct Repository Location

The Octo repository is located at:
**https://github.com/octo-models/octo**

## Installation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/octo-models/octo.git
```

### Step 2: Navigate to the Directory
```bash
cd octo
```

### Step 3: Install Octo
```bash
pip install -e .
```

### Step 4: Install Additional Dependencies
```bash
# Install PyTorch and image processing
pip install torch Pillow

# Install JAX/Flax (required by Octo)
# For CPU-only:
pip install 'jax[cpu]' flax

# OR for GPU support (CUDA 12):
# pip install 'jax[cuda12]' flax
```

### Step 5: Return to Your Project
```bash
cd /home/christoszacharia350/oxe-viz
```

### Step 6: Run Evaluation
```bash
make evaluate-all
```

### Option 3: Use Alternative Models
The evaluation framework supports multiple models. You can use:
- **RT-1** (Robotic Transformer 1)
- **RT-2** (Vision-Language-Action)
- **OpenVLA** (Open Vision-Language-Action)

To use a different model, modify the Makefile or run:
```bash
PYTHONPATH=src python -c "from oxe_viz.evaluate_models import evaluate_all_datasets; evaluate_all_datasets('rt-1', 'val', max_trajectories_per_dataset=5, output_file='evaluation_results.json')"
```

### Option 4: Skip Model Evaluation
If you only need to visualize the data, you can skip evaluation:
```bash
make visualize-all
```

## Current Status

The evaluation framework is ready, but requires the Octo library to be installed. Once you have access to the Octo repository or find the correct installation method, follow these steps:

1. Clone the repository
2. Install: `cd octo && pip install -e .`
3. Install dependencies: `pip install torch Pillow`
4. Run evaluation: `make evaluate-all`

## Verification

To check if Octo is installed:
```bash
python -c "import octo; print('Octo installed successfully')"
```

If this command fails, Octo is not installed.

