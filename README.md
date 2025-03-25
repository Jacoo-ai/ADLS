# TensorRT-MASE: Multi-Precision Quantization & Meta-Learning Recommendation

This project supports quantization, transformation, and intelligent configuration recommendation for deep learning models. Experiments have been tested on RTX4060 (Laptop), A100, and T4. The dictionary of this project is `docs/tutorials/proj` directory and you can get started using `docs/tutorials/proj/tensorRT_meta.ipynb`

## Project Overview

**TensorRT-MASE** is an extensible framework designed to enable **multi-precision quantization**, **model transformation**, and **meta-learning–based recommendation** for deep learning deployment optimization. It provides tools for:

- Performing **int8/fp16/fp32** quantization with TensorRT.
- Automating transformation experiments across different quantization strategies and batch sizes.
- Collecting runtime metrics including latency, accuracy, and energy consumption.
- Training a lightweight **multi-head neural network** to learn optimal configurations from historical experiments.
- Recommending model, batch size, and quant method given user-defined performance targets.

**Key Features:**

- Modular passes and flexible TOML-based strategy definition.
- Safe quantization pipeline compatible with the latest TensorRT releases.
- Supports automatic exploration + visualization of performance trade-offs.
- Includes an intelligent recommendation system for optimal deployment settings.

This makes TensorRT-MASE ideal for **hardware-aware deployment**, **quantization benchmarking**, and **automated model search** in production and research scenarios.

## What's New

### 1. Updated Quantization Workflow

- Removed reliance on fake quant + calibration flags (deprecated in TensorRT).
- Introduced **new INT8 calibration class** based on direct data feeding and layer-wise quantization.
- Visualized default precision for debugging.
- Replaced deprecated `max_workspace_size` with `memory_pool_limit`.
- Backward-compatible with previous transform training process.

### 2. New INT8 Calibration Function

- Supports sequential feeding with internal iterator.
- Limits calibration to a fixed number of batches (`nCalibration`).
- Fully disables INT8 cache reading/writing for **fresh recalibration every run**.

### 3. Safe Deepcopy Function

- Only deepcopies the FX Graph; skips copying CUDA tensors to avoid memory collapse.
- Performs light copy of the MaseGraph to preserve structure safely.

### 4. Registered New Models

- `vgg7_cifar`, `resnet_18`, `resnet_50` model registered in the CHOP library.

### 5. Configurable Training via TOML Files

- Model training and quantization strategies are driven by TOML configuration.
- `num_workers` can be disabled if SHM (Shared Memory) issues occur.
- Users can create custom `.toml` files for strategy search tailored to their task.

### 6. Automation via `src/chop/actions/transmeta.py`

- Supports:
  - Searching over batch size and quantization strategies.
  - Collecting metrics: latency, accuracy, and energy consumption.
  - Exporting results to a CSV file.
  - Generating latency vs. accuracy tradeoff plots.
- Designed for experimentation only — no model weights or transformed graphs are saved.
- Compatible with the legacy `transform_module` structure.

### 7. Meta-Learning Recommendation

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

## Quick Start

### Environment Set Up

To set up the environment,  `Python >= 3.11.9` is required. Then, run the following command to install pycuda, tensorrt and pytorch-quantization.

```
cd ADLS
pip install -e.
pip install --no-cache-dir --index-url https://pypi.nvidia.com pytorch-quantization
pip install pycuda
python3 -m pip install --upgrade pip
python3 -m pip install wheel
python3 -m pip install --upgrade tensorrt
```

### Get Started

After setting up the environment, you can get started using `docs/tutorials/proj/tensorRT_meta.ipynb`
