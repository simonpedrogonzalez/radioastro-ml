# Minimal ResNet-18 and DINOv2 plan

## Files

| File | Responsibility | Target size |
| --- | --- | ---: |
| `ml/nn_common.py` | load/preprocess data, train, predict, evaluate, save artifacts | ≤250 lines |
| `ml/resnet18.py` | ImageNet ResNet-18 construction and trainable-depth policy | ≤70 lines |
| `ml/dinov2.py` | DINOv2-S/14 adapter, head, and freeze policy | ≤70 lines |
| `ml/run_nn_experiments.py` | job list, resume, progress, aggregate report | ≤180 lines |

Reuse `ml.task_evaluation.evaluate_task` and the plotting/table helpers in
`ml.report`. Replace `ml/tests/test_cnn.py` with focused tests for the new code.

## Data

- Read `dataset.json` with `FitsSimulationDataset`, preserving source-level
  train/validation partitions and labels supplied by the dataset.
- Validate finite aligned `(dirty, clean, residual, psf)` planes and resize the
  full field to `224×224`.
- Form native-grid parity channels from residual `R` using the phase centre:
  `E=(R+J)/2`, `O=(R-J)/2`; resize after deriving them.
- Fit one pixel mean/std per included channel from training samples only. Store
  the statistics in every checkpoint and apply them to validation.
- Raw-product ablations retain four ordered slots and set excluded normalized
  slots to zero. DINOv2 always receives these four slots. ResNet parity uses
  `(E,O)`; its raw-product runs use the four slots.
- Training augmentation is a uniformly sampled `torch.rot90(..., k=0..3)`.

## Models

### ResNet-18

- Load `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`;
  replace `fc` with `Linear(512, K)`.
- Raw-product runs replace `conv1` with a four-channel convolution initialized
  from the three pretrained channel kernels plus their mean, scaled by `3/4`.
  Parity runs use two pretrained channel kernels scaled by `3/2`.
- Train modes: `head` (`fc`), `last` (`layer4[-1] + fc`), and `all`.
- Freeze BatchNorm affine parameters and running statistics in `head` and
  `last`; use ordinary train mode for `all`.

### DINOv2-S/14

- Load the official backbone with
  `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`; pin the
  repository revision in run configuration. ViT-S/14 emits a 384-dimensional
  class token for `224×224` input.
- Model: `Conv2d(4,3,kernel_size=1) → DINOv2-S/14 → Linear(384,K)`.
- Train modes: `linear` (adapter + head) and `last` (adapter + head + final
  transformer block). Keep the remaining backbone in evaluation mode.

## Training

- Weighted cross-entropy with `weight_c=N/(K*n_c)`, AdamW, batch size 16, and
  seeds `42,43,44`; device order CUDA, MPS, CPU.
- Base optimizer groups: head/adapter `1e-3`, ResNet last block `1e-4`, ResNet
  full backbone `1e-5`, DINOv2 last block `1e-5`; weight decay `1e-4`.
- Base duration: 30 epochs, patience 5. Schedule jobs vary one factor at a time:
  LR multiplier `{1/3,3}`, five warmup epochs, or max duration `{20,40}`.
- Select the checkpoint by class-balanced validation cross-entropy, using the
  training-derived class weights for both train and validation loss. Early-stop
  patience begins after warmup.
- The focused `residual100` diagnostic crosses residual-only input with every
  ResNet train mode, runs all 100 epochs beginning with five warmup epochs and
  without early stopping, and retains the minimum balanced-validation-loss
  checkpoint.
- On all four products, run every train mode and the schedule jobs. Channel jobs
  use ResNet `last` and DINOv2 `linear`: all products, every leave-one-out set,
  every individual product, plus `(E,O)` for ResNet. Each job has three seeds.

## Runs and report

- Stable job ID: backbone, explicit trainable scope, channels, schedule, seed.
  Scope names are ResNet `head_only`, `head_and_last`, `all` and DINOv2
  `head_adapter`, `head_last_adapter`; the internal `mode` remains in config.
  Write each job
  to `ml/runs/nn/<job-id>/` with `checkpoint.pt`, `config.json`, `history.json`,
  `results.json`, and `predictions.csv`. A complete matching job is resumable.
- Evaluate probabilities with `evaluate_task`: main, clean/corrupted detection,
  conditional error identification, per-class metrics, strength cohorts, and
  confusion matrix.
- `run_nn_experiments.py` writes job state atomically, prints epoch/job progress,
  and rebuilds `ml/runs/nn/report.qmd` and `report.html` after every three jobs
  and at exit.
- The combined report shows completed/pending/failed jobs, loss curves, per-run
  metrics, mean ± sample SD over three seeds, 3×5 strength plots with SD bands,
  confusion matrices, trainable parameters, and runtimes.

## Replacement and checks

- Remove `ml/cnn.py`, `ml/cnn_report.py`, and `ml/ablations.py` after the new
  smoke run reproduces data loading, evaluation, checkpointing, and reporting.
- Tests cover training-only normalization, parity geometry, joint rotation,
  channel zeroing, trainable parameter sets, frozen-state stability, one optimizer
  step per backbone, resume behavior, metric schema, and incremental HTML output.
- Smoke one epoch for every train mode, then run one complete three-seed group
  and verify that saved probabilities exactly reproduce `results.json`.

Official DINOv2 interface: <https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md>.
