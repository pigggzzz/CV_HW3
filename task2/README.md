# CALVIN ACT Cross-Environment Generalization

This package implements the homework pipeline for LeRobot ACT on `xiaoma26/calvin-lerobot`:

1. Basic policy training on environment B (`splitB`).
2. Joint multi-environment training on environments A/B/C (`splitA + splitB + splitC`).
3. Zero-shot cross-environment evaluation on environment D (`splitD`).

The code is intentionally organized so that data preparation is deterministic and does **not** produce recursive `splitA_old_old...` folders.

## Key design

The package uses two isolated data areas under `--data-dir`:

```text
<data-dir>/calvin_act_work/raw/       # raw Hugging Face snapshot, never modified by conversion
<data-dir>/calvin_act_work/prepared/  # converted and ACT-compatible LeRobot v3 roots
```

Prepared roots are fixed:

```text
prepared/splitA
prepared/splitB
prepared/splitC
prepared/splitD
prepared/joint_ABC
```

The v2.1 to v3.0 conversion runs only inside a temporary working directory, then the cleaned result is moved to `prepared/splitX`. Converter backup folders are deleted inside the temporary area and are never recursively scanned.

The schema fixer standardizes CALVIN fields for LeRobot ACT:

```text
actions / rel_actions / action-like key -> action
state / robot_obs                        -> observation.state
image                                    -> observation.images.image
```

The zero-shot evaluator uses inference mode `predict_action_chunk()` and computes offline normalized action L1 error against the dataset action chunks. It does not call `policy.forward()` because ACT's training forward computes the VAE/KL term and can fail in eval-time batches when VAE latent parameters are `None`.

## Installation

```bash
conda activate ACT
cd /home/lama/task2/calvin_act_crossenv_final
pip install -e .
pip install jsonlines tqdm wandb huggingface_hub pyarrow pandas safetensors
```

Make sure `lerobot-train` is available:

```bash
which lerobot-train
```

## One-time inspection / preparation

```bash
python -m crossenv_act inspect \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --cuda-id 0
```

This downloads raw data, prepares v3 ACT-compatible split roots, merges A/B/C, and writes:

```text
/home/lama/task2/output/_metadata/prepared_dataset_report.json
```

To rebuild the prepared data from scratch without touching the raw snapshot:

```bash
python -m crossenv_act inspect \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --force-prepare
```

To redownload raw data too:

```bash
python -m crossenv_act inspect \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --force-download \
  --force-prepare
```

## Task 1: basic policy on environment B

```bash
bash scripts/train_basic.sh \
  --cuda-id 0 \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --wandb-mode online \
  --overwrite-output
```

The run directory is:

```text
/home/lama/task2/output/act_env_b
```

## Task 2: joint policy on environments A/B/C

```bash
bash scripts/train_joint.sh \
  --cuda-id 0 \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --wandb-mode online \
  --overwrite-output
```

The run directory is:

```text
/home/lama/task2/output/act_env_abc
```

## Task 3: zero-shot evaluation on environment D

```bash
bash scripts/eval_zero_shot.sh \
  --cuda-id 0 \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --wandb-mode online
```

Results are saved to:

```text
/home/lama/task2/output/zero_shot_env_d/offline_eval_results.json
/home/lama/task2/output/zero_shot_env_d/offline_eval_results.csv
```

For a quick smoke test:

```bash
bash scripts/eval_zero_shot.sh \
  --cuda-id 0 \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --wandb-mode offline \
  --max-batches 5
```

## Debugging checks

Check prepared schema reports:

```bash
find /home/lama/task2/data/calvin_act_work/prepared -name act_schema_report.json -print
```

Check that no recursive backup chain is being used:

```bash
find /home/lama/task2/data/calvin_act_work/prepared -maxdepth 2 -type d | grep '_old' || true
```

The expected output is empty.
