# 🚚 Improving Cross-Problem Vehicle Routing with Locally Augmented Preferences and Representation Disentanglement

[License: MIT](https://opensource.org/licenses/MIT)
[Python 3.10](https://www.python.org/downloads/)
[PyTorch 2.0](https://pytorch.org/)
[CUDA 11.8](https://developer.nvidia.com/cuda-toolkit)
[Docker](https://www.docker.com/)

This is the official repository containing the code of the paper *Improving Cross-Problem Vehicle Routing with Locally Augmented Preferences and Representation Disentanglement*.

---

## 🎯 Overview

This repository implements a **multi-task neural solver** for 16 vehicle routing variants. The two core contributions of the paper are:

- 📈 **POLAR (Preference Optimization with Local Augmented Refinement)** — extends PO with a local-search refinement step during training, so the policy learns from both neural and locally refined solution pairs
- 🧩 **PLE (Progressive Layered Extraction)** — encoder that disentangles shared and task-specific representations across VRP variants

We also implement **AILS-II** in `/search` (Maximo et al., 2024). At inference, users can optionally refine neural solutions via `tester_params.use_refinement: true`. AILS-II was originally proposed for the CVRP; here it is extended to the multi-task VRP setting.

---

## 🐳 Installation (Docker)

### Prerequisites (host machine)

1. [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU training)

### Build & run

```bash
# Clone
git clone https://github.com/AJ-Correa/Routing-POLAR.git
cd Routing-POLAR

# Build image (first run may take 10–20 min)
docker build -t routing-polar:latest .

# Verify GPU inside the container
docker run --rm --gpus all routing-polar:latest python -c "import torch; print(torch.cuda.get_device_name())"

# Interactive shell (mount data + checkpoints)
docker run --rm -it --gpus all \
  --user $(id -u):$(id -g) \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/result:/app/result" \
  routing-polar:latest bash
```

All commands below assume you are inside the container at `/app`, or prefix them with:

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/result:/app/result" \
  routing-polar:latest
```

---

## 📁 Repository Structure

```
Routing-POLAR/
├── 📄 run.py                  # Main entry point (train / fine-tune / test)
├── 📄 config.yaml             # All hyperparameters
├── 📄 trainer.py              # Training loop on 16 variants
├── 📄 tuner.py                # Fine-tuning on unseen variants (mb / md / both)
├── 📄 tester.py               # Evaluation on synthetic and CVRPLib instances
├── 📂 models/                 # Transformer encoder, decoder, layers, helpers...
├── 📂 envs/
│   ├── 📂mtvrp/               # Single-depot multi-task environment
│   └── 📂mtdvrp/              # Multi-depot environment
├── 📂 search/                 # PyVRP local search + AILS-II (`search/search.py`)
│   └── 📂cython_heuristics/   # Cython distance heuristics (built in Dockerfile)
├── 📂 neural_baselines/       # MTPOMO, MVMoE, RouteFinder
├── 📂 utils/                  # Metrics, helpers
├── 📂 data/                   # Test datasets
├── 📂 result/                 # Checkpoints & logs
├── 📄 Dockerfile
├── 📄 requirements.txt
└── 📄 .dockerignore
```

---

## 🔬 Neural Baselines (`neural_baselines/`)

The `neural_baselines/` folder provides alternative architectures that plug into the same training and evaluation pipeline:


| Baseline        | Path                            |
| --------------- | ------------------------------- |
| **MTPOMO**      | `neural_baselines/mtpomo/`      |
| **MVMoE**       | `neural_baselines/mvmoe/`       |
| **RouteFinder** | `neural_baselines/routefinder/` |


Each baseline exposes a `VRPModel` class compatible with `trainer.py` and `tuner.py`. You can:

- Enable the **PLE encoder** via `use_ple: true` in `config.yaml` (each baseline ships its own `encoder_ple.py`)
- Train with **POLAR** — the locally augmented PO loss (`trainer_params.loss_function: 'po'` + `use_ls: true`)

To run a baseline, replace the model import in `trainer.py` and/or `tuner.py`:

```python
# Default (paper model)
from models.model import VRPModel

# Baselines — uncomment one:
# from neural_baselines.mtpomo.model import VRPModel
# from neural_baselines.mvmoe.model import VRPModel
# from neural_baselines.routefinder.model import VRPModel
```

---

## 🧠 Configuration (`config.yaml`)

All hyperparameters live in `config.yaml`. CLI flags override a subset (problem size, batch size, mode flags, checkpoint paths).

### 🧠 Model Parameters (`model_params`)


| Parameter           | Description                                                  | Default   |
| ------------------- | ------------------------------------------------------------ | --------- |
| `embedding_dim`     | Hidden dimension of node embeddings                          | `128`     |
| `encoder_layer_num` | Number of transformer encoder layers                         | `6`       |
| `qkv_dim`           | Dimension per attention head (Q/K/V)                         | `16`      |
| `head_num`          | Number of attention heads                                    | `8`       |
| `ff_hidden_dim`     | Feed-forward hidden dimension                                | `512`     |
| `ffd`               | FFN type: `'ffd'` (standard) or `'siglu'` (ParallelGatedMLP) | `'siglu'` |
| `norm_type`         | Normalization: `'rms'`, `'layer'`, `'instance'`, `'none'`    | `'rms'`   |
| `use_sparse`        | Sparse attention mode (`'topk'`, etc.)                       | `'topk'`  |
| `p_num`             | Number of constraint prompt tokens (O, L, TW, B, MB, MD)     | `6`       |
| `logit_clipping`    | Decoder logit clipping value                                 | `10`      |
| `use_ple`           | Enable Progressive Layer Extraction encoder                  | `true`    |
| `use_film`          | Enable FiLM task conditioning                                | `true`    |
| `use_reld`          | Enable ReLD                                                  | `true`    |
| `use_rope`          | Enable 2D Rotary Positional Embeddings                       | `true`    |


### 🔍 Tester Parameters (`tester_params`)


| Parameter        | Description                                                                                                                 | Default        |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `use_refinement` | Enable AILS-II iterative refinement during inference                                                                        | `false`        |
| `ls_nb_granular` | LS neighbourhood granularity (φ in AILS-II)                                                                                 | `30`           |
| `stop_condition` | Stopping criterion: `'iterations'` or `'time'`                                                                              | `'iterations'` |
| `num_iters`      | Max ILS iterations (when `stop_condition='iterations'`)                                                                     | `250`          |
| `num_seconds`    | Wall-clock budget in seconds (when `stop_condition='time'`); `null` → auto formula in CVRPLIB tests (num_nodes * 240 / 100) | `null`         |
| `dmax`           | Initial target edge-distance between reference and post-LS solution                                                         | `10`           |
| `dmin`           | Final target edge-distance                                                                                                  | `5`            |
| `gamma`          | Ω adjustment interval (iterations)                                                                                          | `30`           |
| `eta_min`        | Minimum acceptance relaxation η                                                                                             | `0.01`         |


### 🏋️ Trainer Parameters (`trainer_params`)


| Parameter             | Description                                                   | Default  |
| --------------------- | ------------------------------------------------------------- | -------- |
| `epochs`              | Total training epochs                                         | `300`    |
| `train_episodes`      | Training instances per epoch                                  | `100000` |
| `model_save_interval` | Save checkpoint every N epochs                                | `10`     |
| `use_amp`             | Automatic mixed precision (BF16 on Ampere+, else FP16)        | `true`   |
| `loss_function`       | `'po'` (Preference Optimization) or `'rl'` (REINFORCE)        | `'po'`   |
| `po_alpha`            | PO temperature on log-likelihoods                             | `0.05`   |
| `po_B`                | Solutions per instance in PO batch (`null` = all POMO starts) | `null`   |
| `use_ls`              | Enable PyVRP local search for locally augmented PO            | `true`   |
| `ls_start_epoch`      | Epoch after which LS is activated                             | `250`    |
| `ls_nb_granular`      | Neighbourhood granularity for training LS                     | `20`     |


### 🎛️ Optimizer Parameters (`optimizer_params`)


| Parameter                | Description                         | Default         |
| ------------------------ | ----------------------------------- | --------------- |
| `optimizer.lr`           | Learning rate (AdamW)               | `3e-4`          |
| `optimizer.weight_decay` | L2 regularization                   | `1e-6`          |
| `scheduler.name`         | LR scheduler (`'MultiStepLR'` only) | `'MultiStepLR'` |
| `scheduler.milestones`   | Epochs to reduce LR                 | `[270, 295]`    |
| `scheduler.gamma`        | LR decay factor                     | `0.1`           |


### 🔧 Fine-Tuning Parameters (`tuner_params`)

Fine-tuning on **unseen** variants, following Berto et al. (2025). Activated with the CLI flag `--tune`.


| Parameter             | Description                                                                         | Default |
| --------------------- | ----------------------------------------------------------------------------------- | ------- |
| `epochs`              | Fine-tuning epochs                                                                  | `10`    |
| `train_episodes`      | Instances per epoch                                                                 | `10000` |
| `model_save_interval` | Save tuned checkpoint every N epochs                                                | `10`    |
| `test_interval`       | Run variant tests every N epochs                                                    | `5`     |
| `use_amp`             | Automatic mixed precision                                                           | `true`  |
| `loss_function`       | `'po'` or `'rl'`                                                                    | `'po'`  |
| `po_alpha`            | PO temperature                                                                      | `0.05`  |
| `po_B`                | Solutions per instance (`null` = all POMO)                                          | `null`  |
| `use_ls`              | Local search during fine-tuning                                                     | `false` |
| `ls_start_epoch`      | Fine-tuning epoch to activate LS                                                    | `1`     |
| `ls_nb_granular`      | LS neighbourhood granularity                                                        | `20`    |
| `variant_present`     | Variant set: `'mb'` (8 mixed backhaul), `'md'` (16 multi-depot), `'both'` (8 MD+MB) | `'md'`  |


### 🎛️ Fine-Tuning Optimizer (`tuner_optimizer_params`)


| Parameter                | Description                     | Default |
| ------------------------ | ------------------------------- | ------- |
| `optimizer.lr`           | Fine-tuning learning rate       | `1e-4`  |
| `optimizer.weight_decay` | Weight decay                    | `1e-6`  |
| `scheduler.milestones`   | Fine-tuning epochs to reduce LR | `[8]`   |
| `scheduler.gamma`        | LR decay factor                 | `0.1`   |


### 🌍 Environment Parameters (`env`)


| Parameter                         | Description                                               | Default     |
| --------------------------------- | --------------------------------------------------------- | ----------- |
| `generator_params.num_loc`        | Customer count — overridden by `--n_size` (`50` or `100`) | `50`        |
| `generator_params.variant_preset` | Training variants (`'all'` = 16 variants)                 | `'all'`     |
| `test_epoch`                      | Extra epochs that always trigger testing                  | `[50, 150]` |
| `test_interval`                   | Test every N epochs (with `--test`)                       | `10`        |
| `test_episodes`                   | Test instances per evaluation                             | `1000`      |
| `test_batch_size`                 | Test batch size                                           | `100`       |
| `data_dir`                        | Root directory for datasets                               | `'./data'`  |


---

## 🚀 Command Line Arguments


| Argument        | Description                                                  | Default                     |
| --------------- | ------------------------------------------------------------ | --------------------------- |
| `--n_size`      | Problem size (`50` or `100`)                                 | `50`                        |
| `--batch_size`  | Training batch size per GPU                                  | `256`                       |
| `--num_workers` | DataLoader worker processes                                  | `0`                         |
| `--seed`        | Random seed                                                  | `7`                         |
| `--resume`      | Resume from checkpoint                                       | `false`                     |
| `--epoch`       | Checkpoint epoch to load (required with `--resume`)          | —                           |
| `--path_id`     | Checkpoint folder under `result/` (required with `--resume`) | —                           |
| `--test`        | Enable periodic testing during training/fine-tuning          | `false`                     |
| `--test_only`   | Run testing only (no training)                               | `false`                     |
| `--test_lib`    | Evaluate on CVRPLIB benchmarks only                          | `false`                     |
| `--skip`        | Quick smoke test (few steps/batches)                         | `false`                     |
| `--tune`        | Fine-tuning mode (unseen variants)                           | `false`                     |
| `--variant`     | Fine-tuning set: `mb`, `md`, or `both`                       | `null` (uses `config.yaml`) |
| `--wandb`       | Weights & Biases run ID (`''` = disabled)                    | `''`                        |


**Checkpoint path convention:** `--path_id` is relative to `result/`. Example:

```text
--path_id "POLAR/n=50/2026-0515-1225"  →  result/POLAR/n=50/2026-0515-1225/checkpoint-{epoch}.pt
```

Replace the folder name with your own run directory under `result/`.

**Multi-GPU:** launch with `torchrun` (sets `LOCAL_RANK` / `WORLD_SIZE` automatically):

```bash
torchrun --nproc_per_node=2 run.py --n_size 50 --batch_size 128
```

---

## 📦 Dataset

Place benchmark data under `data/` before training or testing:

```
data/
├── cvrp/           # Synthetic single-depot test sets
├── ovrp/
├── …               # One folder per variant
├── lib_data/       # CVRPLib instances (A, B, F, M, P, X, …)
└── md*/            # Multi-depot folders
```

Synthetic evaluation sets are generated per variant under `data/<variant>/`. CVRPLIB instances for `--test_lib` live in `data/lib_data/`.

---

## 🏋️ Training

Train on all 16 VRP variants with Preference Optimization and late-activation local search (epoch 250+).

### VRP with 50 nodes

```bash
python run.py --n_size 50 --batch_size 256 --test
```

### VRP with 100 nodes

```bash
python run.py --n_size 100 --batch_size 256 --test
```

> ⚠️ `**--test**` runs evaluation every `test_interval` epochs (default 10) plus at epochs listed in `test_epoch` (`[50, 150]`).

### Recommended `config.yaml` snippet

```yaml
model_params:
  use_ple: true

trainer_params:
  loss_function: 'po'
  po_alpha: 0.05
  use_ls: true
  ls_start_epoch: 250
```

### Docker example

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/result:/app/result" \
  routing-polar:latest \
  python run.py --n_size 50 --batch_size 256 --test
```

---

## 🔁 Reproducing Paper Experiments

Pre-trained checkpoints (epoch 300) are available under `result/POLAR/`:

### Evaluate on synthetic benchmarks

```bash
# n = 50 (example path_id — use your run folder if different)
python run.py --n_size 50 --test --test_only \
  --resume --epoch 300 --path_id "POLAR/n=50/2026-0515-1225"

# n = 100
python run.py --n_size 100 --test --test_only \
  --resume --epoch 300 --path_id "POLAR/n=100/2026-0517-1400"
```

---

## 🔧 Fine-Tuning

Fine-tune a pretrained checkpoint on **unseen** variants. Requires `--resume`.

### Multi-depot variants (`md`, 16 types)

```bash
python run.py --tune --variant md --n_size 50 \
  --resume --epoch 300 --path_id "POLAR/n=50/2026-0515-1225" --test
```

### Mixed backhaul (`mb`, 8 types)

```bash
python run.py --tune --variant mb --n_size 50 \
  --resume --epoch 300 --path_id "POLAR/n=50/2026-0515-1225" --test
```

### Multi-depot + mixed backhaul (`both`, 8 types)

```bash
python run.py --tune --variant both --n_size 100 \
  --resume --epoch 300 --path_id "POLAR/n=100/2026-0517-1400" --test
```

Tuned weights are saved to `result/<timestamp>/tuned-{variant}-{epoch}.pt`.

---

## 🧪 Testing / Inference

### Synthetic benchmarks (all 16 training variants)

```bash
python run.py --n_size 50 --test --test_only \
  --resume --epoch 300 --path_id "POLAR/n=50/2026-0515-1225"
```

### Fine-tuning benchmarks (works with `md`, `mb`or `both`)

```bash
python run.py --tune --variant md --test_only \
  --resume --epoch 300 --path_id "POLAR/n=50/2026-0515-1225"
```

### CVRPLIB benchmark (`--test_lib`)

```bash
python run.py --n_size 100 --test_lib \
  --resume --epoch 300 --path_id "POLAR/n=100/2026-0517-1400"
```

Reads instances from `data/lib_data/`.

### AILS-II refinement (optional)

AILS-II (Maximo et al., 2024) is implemented in `search/search.py`. Although originally designed for the CVRP, we extend it to the multi-task VRP setting. Users can optionally refine neural solutions at inference by enabling:

```yaml
tester_params:
  use_refinement: true
  stop_condition: 'iterations'   # or 'time'
  num_iters: 250    # if stop_condition=='iterations'
  num_seconds: 300  # if stop_condition=='time'
```

Then run any test command above.

---

## ⏯️ Resume Interrupted Training

```bash
python run.py --n_size 50 --batch_size 256 --test \
  --resume --epoch 150 --path_id "POLAR/n=50/2026-0515-1225"
```

---

## 📚 References (AILS-II)

Maximo, V. R., Cordeau, J.-F., & Nascimento, M. C. V. (2024). AILS-II: An Adaptive Iterated Local Search Heuristic for the Large-Scale Capacitated Vehicle Routing Problem. *INFORMS Journal on Computing*, 36(4), 974–986. [https://doi.org/10.1287/ijoc.2023.0106](https://doi.org/10.1287/ijoc.2023.0106)

---

## 📄 License

This project is licensed under the MIT License.