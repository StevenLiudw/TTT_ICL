# TTT_ICL on PILE(Uncopy-right subset --00 partition) README
The script supports GitHub, Enron, and Math datasets.

The main operations include:

- **Launching Data Servers:** Distributing dataset inputs via a server.
- **Evaluating and Training:** Running TTTLM which:
  - Evaluates baseline performance.
  - Retrieves nearest neighbor examples.
  - Fine-tunes the model on these neighbors.
  - Performs inference via neighbors-based and random in-context learning (ICL).
  - Measures FLOPs and other resource usage.
  
The evaluation results and cost metrics are saved to files in the specified results directory.

---

## 1. Environment Setup

Create the Conda environment using the provided `ttt_env.yaml`:

```bash
conda env create -f ttt_env.yaml
conda activate ttt_nn
```

---

## 2. Unified Script for Three Datasets

All three datasets use the same script. Use the following commands to process each dataset:

### GitHub Dataset

```bash
python3 code/pile_server.py --address_path servers/addresses.txt --data_file 00_github.jsonl --num_servers 6
CUDA_VISIBLE_DEVICES= CUDA_LAUNCH_BLOCKING=1 python3 code/github.py --address_path servers/addresses.txt --results_dir results
```

### Enron Dataset

```bash
python3 code/pile_server.py --address_path servers/address_enron.txt --data_file 00_enron.jsonl --num_servers 6
CUDA_VISIBLE_DEVICES= CUDA_LAUNCH_BLOCKING=1 python3 code/enron.py --address_path servers/address_enron.txt --results_dir results
```

### Math Dataset

```bash
python3 code/pile_server.py --address_path servers/addresses_math.txt --data_file 00_dm_math.jsonl --num_servers 6
CUDA_VISIBLE_DEVICES= CUDA_LAUNCH_BLOCKING=1 python3 code/dm_math.py --address_path servers/addresses_math.txt --results_dir results
```

---

## 3. What the Code Does & Saved Results

### Code Overview

- **Evaluation Approaches:**  
  Three methods are used:
  - **TTT:** Baseline evaluation, neighbor retrieval, training on neighbors, and re-evaluation.
  - **Neighbors ICL:** Evaluation using a prompt augmented with retrieved neighbors (no training).
  - **Random ICL:** Evaluation using a prompt augmented with random examples (no training).

- **FLOPs Calculation:**  
  FLOPs, MACs, and parameter counts are computed using the `calflops` module.

- **Logging & Aggregation:**  
  Metrics such as total time, training time, retrieval time, test time, and GPU memory usage are logged. Results from multiple trials are aggregated to compute means and variances.

### Saved Results

- **Trial Results File:**  
  For each task (dataset), the script saves a results file (e.g., `github_0.pth`, `enron_0.pth`, `dm_math_0.pth`) in the `results` directory. This file contains:
  - TTT trials data.
  - Neighbors ICL trials data.
  - Random ICL trials data.
  
- **Cost Metrics File:**  
  A separate JSON file (with a `_costs.json` suffix) summarizes cost records per sample, including time metrics, memory usage, and FLOPs.

---

## 4. Customizing Validation/Test Files

The `pile` folder contains several JSONL files, including:

- `test_enron.jsonl`
- `test_ori.jsonl`
- `val.jsonl`
- `val_math.jsonl`
- `test.jsonl`
- `test_math.jsonl`
- `train`
- `val_enron.jsonl`
- `val_ori.jsonl`

By default, the validation (`val.jsonl`) and test (`test.jsonl`) files are configured for the GitHub dataset. If you wish to run evaluations on a different dataset (e.g., Enron or Math), you will need to manually modify the file names in your command or configuration to point to the appropriate JSONL files.

For example, for the Enron dataset, you might change:
- From `val.jsonl` to `val_enron.jsonl`
- From `test.jsonl` to `test_enron.jsonl`

Ensure that your script or configuration points to the correct file names corresponding to your target dataset.
