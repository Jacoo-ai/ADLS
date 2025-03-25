# TensorRT-MASE: Multi-Precision Quantization & Meta-Learning Recommendation

This project supports quantization, transformation, and intelligent configuration recommendation for deep learning models. Experiments have been tested on RTX4060 (Laptop), A100, and T4. Results are saved under the `proj` directory.

## Notebook Overview

### What's New

#### 1. Updated Quantization Workflow

- Removed reliance on fake quant + calibration flags (deprecated in TensorRT).
- Introduced **new INT8 calibration class** based on direct data feeding and layer-wise quantization.
- Visualized default precision for debugging.
- Replaced deprecated `max_workspace_size` with `memory_pool_limit`.
- Backward-compatible with previous QAT training process.

#### 2. New INT8 Calibration Function

- Supports sequential feeding with internal iterator.
- Limits calibration to a fixed number of batches (`nCalibration`).
- Fully disables INT8 cache reading/writing for **fresh recalibration every run**.

#### 3. Safe Deepcopy Function

- Only deepcopies the FX Graph; skips copying CUDA tensors to avoid memory collapse.
- Performs light copy of the MaseGraph to preserve structure safely.

#### 4. Registered New Models

- `VGG_cifar` model registered in the CHOP library.
- `RESNET_50` model registered in the CHOP library.

#### 5. Configurable Training via TOML Files

- Model training and quantization strategies are driven by TOML configuration.
- `num_workers` can be disabled if SHM (Shared Memory) issues occur.
- Users can create custom `.toml` files for strategy search tailored to their task.

#### 6. Automation via `Transmeta`

- Supports:
  - Searching over batch size and quantization strategies.
  - Collecting metrics: latency, accuracy, and energy consumption.
  - Exporting results to a CSV file.
  - Generating latency vs. accuracy tradeoff plots.
- Designed for experimentation only — no model weights or transformed graphs are saved.
- Compatible with the legacy `transform_module` structure.

#### 7. Meta-Learning Recommendation

- Reads the experiment CSV output, applies normalization, and labels data.
- Splits into training and testing sets using PyTorch tensors.
- Trains a **3-head neural network** to predict:
  - Model name
  - Quantization strategy
  - Batch size
- Uses `CrossEntropyLoss`; batch size loss is weighted 10× higher.
- Trains for 500 epochs.
- Accepts user input of target latency, accuracy, and energy to recommend:
  - Best model name
  - Batch size
  - Quantization method
- Evaluation function reports accuracy for each head and full-match rate.

## How to Use This Notebook

### Section 1: Training the Base Model

We recommend training the base model using INT8 calibration as default precision.
Adjust paths according to your file structure and filenames.

### Section 2: Meta-Transform Learning

Uses meta `.toml` files and pre-trained checkpoints from Section 1 to perform full strategy search.
Checkpoints are versioned by date. Output includes performance metrics and graphs.

### Section 3: Meta-Learning

Reads transformation results from the `mase-output` folder.
Users may create new metadata files, specify desired performance constraints (latency, accuracy, energy), and run meta-learning to get optimal quantization recommendations.
Training runs for 500 epochs.

## Environment Requirements

Ensure the following dependencies and system setup:

- **OS:** Ubuntu 20.04.6
- **Python:** 3.11.11
- **PyTorch:** 2.6.0
- **CUDA:** 12.4.99
- **cuDNN:** ubuntu2004-9.8.0_1.0-1_amd64
- **cuda-python:** 12.8.0
- **TensorRT:** 10.9.0.34
- **pip:** 25.0
