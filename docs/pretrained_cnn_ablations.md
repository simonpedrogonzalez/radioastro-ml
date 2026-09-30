# Ablations for the square-only CNN

Status: completed on 2026-09-22 and retained as a historical result. Its runner was superseded by [the minimal ResNet-18/DINOv2 stack](minimal_nn_approaches_plan.md) and removed.

## Run the implemented suite

From the repository root:

```sh
uv run --project ml python -m ml.ablations \
  --dataset collect/experiments/dataset_v1_20260921_all_psf/dataset.json \
  --reference-parity ml/runs/20260922T162116896637Z_resnet18_parity \
  --reference-residual ml/runs/20260922T162316909007Z_resnet18_residual
```

This reuses the completed seed-42 channel runs after checking dataset identity, preprocessing and exact train/validation sample IDs, labels and levels. It runs seven new CNN fits (four channel replications and three matched head-only controls), two parity-feature logistic models, and refits the four/14-feature QA controls. FITS data is loaded once per channel mode. Outputs go to a new timestamped `ml/runs/*_ablations/`, with checkpoints, per-run HTML reports, logistic models/features/predictions, and a combined `report.html`/`summary.json`. Test evaluation is disabled in this runner. The optional pretraining, augmentation, individual-channel and cropping studies below remain deferred.

The standalone CNN also accepts `--head-only`: its second stage keeps only the head trainable while retaining the original two-stage learning rates and stopping rule. Console updates show source-loading progress and each epoch's train/validation loss, balanced accuracy, validation F1, selected epoch and patience count. Training metrics are measured on augmented minibatches; validation is deterministic.

## Completed results

[Combined HTML report](../ml/runs/20260922T191944331945Z_ablations/report.html) · [All metrics and provenance](../ml/runs/20260922T191944331945Z_ablations/summary.json).

Seven new CNN fits and four logistic fits completed; the comparison also includes the two original seed-42 CNNs. All use the same 651 training samples / 73 sources and 143 validation samples / 16 sources. Test was not evaluated. CNN values below are mean ± sample SD across seeds42/43/44; logistic values are single deterministic fits.

| Model | Validation balanced accuracy | Macro F1 |
| --- | ---: | ---: |
| Four QA + two parity features, logreg | **.8581** | .7676 |
| Two parity features only, logreg | .8477 | **.7694** |
| Parity CNN, fine-tuned | .8024 ± .0081 | .7079 ± .0057 |
| Parity CNN, matched head-only | .7416 ± .0210 | .6779 ± .0117 |
| Residual-only CNN, fine-tuned | .5839 ± .0236 | .4699 ± .0204 |
| Original 14 QA features, logreg | .5803 | .4906 |
| Original four QA features, logreg | .5330 | .4339 |

The channel advantage persists across all three seeds. Fine-tuning helps the CNN relative to the matched head-only control, but simple parity features give higher validation scores here. **Parity-feature logreg is the stronger candidate for the next evaluation.** This does not establish that no CNN could improve further, or isolate a benefit from ImageNet pretraining.

Weak-error detection remains limited. At SNR10, the six-feature model detects and correctly types **6/32** examples (18.8%); parity-only logreg reaches **7/32** (21.9%). Their clean false positives are **0/16** and **1/16**, respectively. The fine-tuned parity CNN averages 9.4% detection and 7.3% correct type at this level, with 2.1% mean clean FPR across seeds. The combined report includes all levels and every seed.

These historical correct-type counts use all corrupted samples as the denominator. The report's type column now uses detected corrupted samples as its denominator.

Descriptive paired bootstrap over the 16 validation sources (5,000 draws, seed2026; retaining all variants, averaging the fixed model seeds):

- Parity CNN minus residual CNN: **+21.9 percentage points BA**, interval **[17.6, 26.1]**.
- Fine-tuned parity CNN minus head-only: **+6.1 points**, **[1.9, 10.8]**.
- QA4+parity logreg minus fine-tuned parity CNN: **+5.6 points**, **[3.7, 7.5]**.
- Two-feature logreg minus fine-tuned parity CNN: **+4.5 points**, **[-1.7, 8.7]**.
- Six-feature minus two-feature logreg: **+1.0 point**, **[-2.6, 6.2]**.

These intervals are conditional on the selected validation set and fitted models; they do not correct for validation-based selection or describe variation from new training datasets. In particular, the choice between the two simple parity classifiers is not settled by this small validation set.

Verification: all eight ML tests passed. Saved head-only checkpoints preserve every pretrained backbone parameter and BatchNorm buffer; their five warmup epochs match the corresponding fine-tuned runs. Saved logistic models reproduce their predictions, and their scalers match training-only feature means. All 13 models' metrics were recalculated from saved predictions; sample partitions, exclusions, probabilities and CNN checkpoint selection agree with their artifacts. Reports contain embedded plots. Pretraining/scratch, augmentation, channel removal and crop experiments remain deferred.

## What needs explaining

The existing `[R,E,O]` versus `[R,R,R]` comparison is already a useful representation ablation: same architecture, training recipe, samples and seed; different channels. Here `R` is the residual, `E` its even component and `O` its odd component about the phase centre.

| Seed-42 validation result | Balanced accuracy | Macro F1 |
| --- | ---: | ---: |
| CNN `[R,E,O]` | .8006 | .7018 |
| CNN `[R,R,R]` | .5594 | .4532 |
| Matched 14-feature logreg | .5803 | .4906 |

These results cover 143 validation samples from **16 sources**, trained on 651 samples from 73 sources. They support investigating parity, but do not establish that pretraining, a CNN, or fine-tuning is necessary.

The parity model's 38 errors are all corrupted samples classified as clean; it makes no amp/phase swaps in this validation run. Detection and correct-type rates are both **1/32 at SNR10**, **26/32 at SNR30**, **31/32 at SNR50**, and **31/31 at SNR100**. Clean false positives are 0/16. Residual-only detects most strong errors too, but frequently confuses amp and phase. This suggests two questions: what supplies the type information, and what limits weak-error detection? The absence of type errors in one small validation set is not a guarantee.

Local evidence: [parity results](../ml/runs/20260922T162116896637Z_resnet18_parity/results.json), [residual-only results](../ml/runs/20260922T162316909007Z_resnet18_residual/results.json), and their `history.json` files.

## Smallest useful sequence

### 1. Repeat the existing channel comparison

Run both modes with seeds **43 and 44**, retaining seed 42: four additional CNN fits. Report each paired result and the mean/std across all three seeds. This is replication rather than a new ablation. It establishes whether the large channel difference persists under training randomness; it does not measure variation across new observing sources. Benchmarking research identifies initialization, data sampling and other choices as distinct sources of variation. [Bouthillier et al., MLSys 2021](https://arxiv.org/abs/2103.03098).

The current CLI already supports this, from the repository root:

```sh
for seed in 43 44; do
  for mode in parity residual; do
    uv run --project ml python -m ml.cnn \
      --dataset collect/experiments/dataset_v1_20260921_all_psf/dataset.json \
      --input-mode "$mode" --seed "$seed"
  done
done
```

### 2. Test whether two parity statistics explain the gain

This is the most useful cheap new baseline. From each sample alone, calculate two features on the native paired domain `D = [1:,1:]`:

```text
u = mean_D((E/s)^2)
v = mean_D((O/s)^2)
features = [log1p(u), log1p(v)]
```

Use the existing centre `(128,128)`, E/O equations and residual-annulus MAD scale `s`. Calculate these before clipping, asinh or resizing. The paired domain omits only the unmatched first row/column; it introduces no circular aperture. Do not use a paired uncorrupted reference or injected SNR as a feature.

Fit the existing `StandardScaler -> balanced LogisticRegression(C=1)` to:

- The two parity features alone.
- The current four QA features plus these two features.

Two deterministic, inexpensive fits; fit the scaler on training only. This is a new baseline testing a physical summary, rather than a strict one-component CNN ablation. The hypothesis comes from the training audit's even/odd structure, not from a published guarantee for this dataset.

If these approach the CNN across sources and SNR levels, much of the gain may come from exposing parity. If they lag, the CNN adds value beyond these particular summaries; that would not rule out every simpler classifier.

### 3. Check whether the last-block fine-tuning earns its complexity

The existing parity history already shows **.7219 BA after head-only warmup**, versus **.8006** after fine-tuning. Residual-only selected its warmup checkpoint. These are useful diagnostics, but five head epochs versus up to twenty total epochs mixes the effect of trainable layers with training duration.

For a controlled comparison, use parity inputs and seeds 42/43/44. Repeat the same five-epoch warmup, reload its best checkpoint, then keep the backbone frozen during the second stage. Keep its optimizer reset, head LR `3e-4`, weight decay, maximum 15 extra epochs, patience 5 and checkpoint selection identical. The only intended change is whether the convolutions in `layer4[1]` can update. All BatchNorm parameters and buffers remain frozen in both arms.

This is implemented by `--head-only` and three additional fits. `--finetune-epochs 0` gives only the short warmup comparison; `--warmup-epochs 20` also changes the learning-rate schedule and is not the matched control. Fixed pretrained features versus fine-tuning are the two standard transfer-learning setups. [PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

If the matched head-only model performs similarly, prefer its 1,539 trainable parameters over 4,720,131, subject to weak-error performance and variation across sources.

## Required before claiming that ImageNet pretraining helps

Current results cannot isolate pretraining: both CNN modes use the same ImageNet checkpoint. Transfer benefits must be checked on the target task; Raghu et al. found limited performance benefits on their medical-image tasks, which motivates a control here but does not predict a radio result. [Transfusion, NeurIPS 2019](https://arxiv.org/abs/1902.07208).

The cheapest narrowly interpretable control is **random initialization with the same partial-freezing/training recipe**, on parity inputs at seeds 42/43/44. It measures the value of the pretrained checkpoint within our chosen training budget. It includes pretrained BatchNorm state as part of that checkpoint. A poor result would be unsurprising because most random layers cannot learn; do not describe it as a fully trained from-scratch baseline or proof that ImageNet pretraining is generally necessary.

For the stronger claim “pretraining beats training from scratch,” separately compare pretrained and random ResNet-18 with **all layers trainable in both**, the same explicitly chosen BatchNorm policy, and comparable optimization/tuning budgets. Check convergence rather than assuming our short transfer-learning schedule suffices for scratch. This is a larger follow-up, not required to deploy a useful classifier. Architecture changes such as a small scratch CNN would answer another question as well.

## Lower-priority follow-ups

- **Augmentation:** parity model with D4 rotations/reflections disabled, retrained with the same schedule. Use separate random generators for shuffling and augmentation so disabling augmentation does not change minibatch order. This tests robustness to orientation; do it after the comparisons above if the gain remains unclear.
- **Individual channels:** retrain `[R,E,0]` and `[R,0,O]` only if we need to attribute the benefit to a particular component. Set the omitted channel to physical zero before asinh/mapping/ImageNet normalization. Zeroing a channel only at inference tests sensitivity to an unfamiliar input, not the performance of a model trained without it. Because `R=E+O` on the paired domain, these channels are redundant in raw information; such tests concern useful representation, not independent observations.
- **Weak-error detection:** inspect the existing probabilities by SNR first. Threshold selection or a two-stage detector is a new decision-rule experiment, not evidence about the CNN architecture. Any detection improvement must also report clean false positives; keep the three-class argmax rule fixed in the core ablations.
- **Circular sources/cropping:** remain deferred as requested. To isolate cropping later, first compare full field versus crop on the same square training and validation sources. Adding circular training sources is a separate factor; comparing models trained on different populations cannot isolate the crop effect.

Do not start a large sweep of architectures, annulus radii, clipping limits, optimizers, PSF/clean/dirty inputs or every channel combination. Those are additional model-development questions. The existing audit found PSF identical across a source's variants, so PSF alone cannot distinguish those variants; any future PSF test should address contextual value.

## Comparison rules and stopping point

Keep the existing source partitions, square-support filter, labels, class weights, preprocessing, selection metric and training settings fixed except for the stated factor. Train every ablated configuration afresh. Keep test evaluation disabled throughout development. Source IDs and corruption levels are for grouping and evaluation only.

Report balanced accuracy, macro F1, per-class recall, clean false-positive rate, and detection/correct-type rates at every SNR. Keep the poor SNR10 result visible; a high overall score must not stand in for weak-error detection. Report parameter counts and training time too. For paired uncertainty estimates, resample whole validation sources with all their variants and use the same sampled sources for both models; do not treat the 143 samples or different seeds as independent source observations. With only 16 validation sources, small differences remain inconclusive. Repeated validation-based selection makes these development results; evaluate the locked choice on test afterwards.

**Original recommended immediate budget: four replication fits plus two small parity-feature logreg fits**, followed by the three matched head-only fits. The implemented suite includes all three comparisons. The earlier CNN runs took approximately 77–111 seconds each including preprocessing and final evaluation, before HTML rendering; shared loading reduces repeated I/O in the suite. Pretraining controls remain a next step if we intend to make a claim about transfer learning.
