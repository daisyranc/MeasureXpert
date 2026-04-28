# MeasureXpert

This repository contains the official source code for **[MeasureXpert](https://daisyranc.github.io/MeasureXpert/?v=2)**.

---

## 🛠️ Environment Setup

### 1. Prerequisites: CUDA Requirements
This project utilizes GPU acceleration and requires a compatible NVIDIA GPU with CUDA installed.
* **Tested Version:** The current code and the provided `environment.yml` have been strictly tested and verified on **CUDA 9.0**.

### 2. Installation
This project uses Anaconda to manage Python dependencies. Please use the provided `environment.yml` file for a quick installation.

Open your terminal and run the following commands in the project's root directory:

```bash
# Create the environment using the provided yml file
conda env create -f environment.yml

# Activate the environment (replace 'your_env_name' with the actual name in your yml)
conda activate your_env_name
```

---

## 🚀 Training

The training pipeline consists of two sequential steps. Please ensure your Conda environment is activated before running the scripts.

**Step 1: Initial Training** First, run the initial training script:
```bash
python train_first_step.py
```

**Step 2: Offset Training** Once the first step is complete, run the following script to train the offsets:
```bash
python train_offset.py
```

---

## 🧪 Testing / Demo

Similar to the training process, testing is also divided into two stages for optimal results.

**Step 1: Standard Demo** Run the base demonstration script:
```bash
python demo.py
```

**Step 2: Refinement** After the initial demo finishes, run the refinement script to get the final polished results:
```bash
python demo_refine.py
```
```

