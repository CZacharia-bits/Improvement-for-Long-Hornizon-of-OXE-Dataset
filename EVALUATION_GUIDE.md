# Evaluation Success Guide

This guide explains how to verify that model evaluation ran successfully and how to interpret the results.

## ⚠️ Common Issue: Empty evaluation_results.json

If `evaluation_results.json` contains only `[]` or is empty, it means **all evaluations failed**. This usually happens because:

1. **Missing dependencies**: The Octo model library is not installed
   - **Solution**: Install Octo from source (see "Installation" section below)
   
2. **Dataset access issues**: Cannot connect to GCS or datasets are unavailable
   - **Solution**: Check your GCS authentication and network connection

3. **Model loading errors**: The model checkpoint cannot be downloaded or loaded
   - **Solution**: Check internet connection and disk space (~2GB needed for Octo)

The updated code now saves error information in the JSON file, so check the `failed` section to see what went wrong.

## Running Evaluation

```bash
# Evaluate on single dataset
make evaluate

# Evaluate on all datasets
make evaluate-all
```

## Signs of Success ✅

### 1. **Progress Messages with Checkmarks**

You should see checkmarks (✓) indicating successful steps:

```
============================================================
Evaluating octo on taco_play (val)
============================================================

Loading model: octo...
  Loading Octo model from checkpoint...
  ✓ Octo model loaded successfully
✓ Model loaded successfully

Loading dataset: taco_play...
✓ Dataset loaded successfully

Running evaluation...
  Processed 10 trajectories...

Computing metrics...
✓ Evaluation complete
```

### 2. **Metrics Summary Printed**

After evaluation completes, you'll see a formatted metrics summary:

```
============================================================
Evaluation Results: octo on taco_play
============================================================
Trajectories evaluated: 10
Total steps: 250

Action Prediction Metrics:
  MSE:  0.123456
  MAE:  0.234567
  RMSE: 0.351234

Reward Statistics:
  Mean: 0.8500
  Std:  0.1200
============================================================
```

### 3. **Output File Created** (for `make evaluate-all`)

When evaluating all datasets, results are saved to `evaluation_results.json`:

```bash
ls -lh evaluation_results.json
# Should show the file exists and has content
```

The JSON file contains metrics for each dataset:
```json
[
  {
    "dataset_name": "taco_play",
    "model_name": "octo",
    "num_trajectories": 10,
    "num_steps": 250,
    "action_mse": 0.123456,
    "action_mae": 0.234567,
    "action_rmse": 0.351234,
    ...
  },
  ...
]
```

### 4. **Summary Statistics** (for `make evaluate-all`)

At the end, you'll see a summary:

```
============================================================
SUMMARY: 10 successful, 2 failed
============================================================

✓ Successfully evaluated (10):
  • taco_play
  • berkeley_autolab_ur5
  • bridge
  ...

✗ Failed (2):
  • language_table: No image feature found...
  • kuka: Dataset is empty...
```

## Signs of Failure ❌

### 1. **Error Messages with X Marks**

```
✗ Failed to load model: Octo library not installed...
✗ Failed to load dataset: No TFRecord files found...
```

### 2. **Exceptions or Tracebacks**

Python tracebacks indicate something went wrong:
```
Traceback (most recent call error):
  File "...", line X, in ...
    ...
ImportError: No module named 'octo'
```

### 3. **No Metrics Printed**

If evaluation fails before computing metrics, you won't see the metrics summary.

### 4. **Zero Trajectories**

```
ValueError: No valid trajectories found in dataset_name
```

## Interpreting Metrics

### Action Prediction Metrics

- **MSE (Mean Squared Error)**: Lower is better. Measures average squared difference between predicted and actual actions.
  - Good: < 0.1
  - Moderate: 0.1 - 1.0
  - Poor: > 1.0

- **MAE (Mean Absolute Error)**: Lower is better. Average absolute difference.
  - Good: < 0.2
  - Moderate: 0.2 - 0.5
  - Poor: > 0.5

- **RMSE (Root Mean Squared Error)**: Lower is better. Square root of MSE, in same units as actions.
  - Good: < 0.3
  - Moderate: 0.3 - 1.0
  - Poor: > 1.0

### Reward Statistics

- **Mean Reward**: Higher is better. Average reward across trajectories.
- **Std Reward**: Lower is better (more consistent). Standard deviation of rewards.

## Quick Verification Checklist

After running evaluation, verify:

- [ ] See checkmarks (✓) for model loading and dataset loading
- [ ] See "✓ Evaluation complete" message
- [ ] Metrics summary is printed with numbers (not all zeros)
- [ ] `num_trajectories > 0` and `num_steps > 0`
- [ ] For `evaluate-all`: `evaluation_results.json` file exists
- [ ] No error messages or exceptions
- [ ] Summary shows successful datasets (for `evaluate-all`)

## Common Issues

### Issue: "Octo library not installed" or Empty evaluation_results.json
**Solution**: 
1. Octo must be installed from source (it's not on PyPI):
   ```bash
   git clone https://github.com/rail-berkeley/octo.git
   cd octo
   pip install -e .
   ```
2. Install additional dependencies:
   ```bash
   pip install torch Pillow
   ```
3. Note: First run will download ~2GB model weights automatically

If `evaluation_results.json` is empty (`[]`), check the terminal output for error messages. The updated code now saves error information in the JSON file under the `failed` key.

### Issue: "No TFRecord files found"
**Solution**: Check GCS access and dataset name spelling

### Issue: "No valid trajectories found"
**Solution**: Dataset might be empty or have no action data

### Issue: All metrics are zero
**Solution**: Model might not be predicting correctly, or predictions aren't being computed

## Example Successful Output

```
============================================================
Evaluating octo on taco_play (val)
============================================================

Loading model: octo...
  Loading Octo model from checkpoint...
  ✓ Octo model loaded successfully
✓ Model loaded successfully

Loading dataset: taco_play...
✓ Dataset loaded successfully

Running evaluation...
  Processed 10 trajectories...

Computing metrics...
✓ Evaluation complete

============================================================
Evaluation Results: octo on taco_play
============================================================
Trajectories evaluated: 10
Total steps: 250

Action Prediction Metrics:
  MSE:  0.045123
  MAE:  0.189456
  RMSE: 0.212567

Reward Statistics:
  Mean: 0.9200
  Std:  0.0800
============================================================
```

This indicates a **successful evaluation**! ✅


