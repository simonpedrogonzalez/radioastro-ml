# Small pretrained CNN experiment for dataset v1

Status: implementation specification, 2026-09-22. No CNN was trained during research.

Research note: [Pretrained CNN for Constant Antenna Corruption Classification](</Users/u1528314/Documents/Obsidian Vault/Notes/Pretrained CNN for Constant Antenna Corruption Classification.md>).

## Scope and decision

Implement **torchvision ResNet-18 with `ResNet18_Weights.IMAGENET1K_V1`** for `none=0`, `amp=1`, `phase=2`. Use signed residual/symmetric/antisymmetric channels, and one control using the residual repeated three times. This tests whether spatial structure adds information beyond QA scalars. It is specific to the centred-point-source, constant-one-antenna experiment.

Use the existing Python 3.12 `ml` uv environment; add compatible `torchvision` with `uv add --project ml torchvision` and commit the resulting local dependency/lock changes. Keep `.venv`, weight downloads (`TORCH_HOME=ml/.cache/torch`), and outputs local and ignored. Do not change CASA dependencies, FITS products, label rules, or partitions.

Architecture alternatives were checked: MobileNetV3-Small is cheaper (0.06 versus ResNet-18's 1.81 published GFLOPs at 224px), and EfficientNet-B0 is intermediate (0.39). ResNet-18 is selected for its direct partial-freezing implementation, not demonstrated superiority on this dataset. Use MobileNet only as a separately reviewed fallback if a measured epoch is too expensive.

## Evidence used to fix the design

Read-only research selected index paths by `TRAIN_IDS` before opening sample manifests/products; validation/test products were not inspected. Dataset: `collect/experiments/dataset_v1_20260921_all_psf/dataset.json`.

| Training audit | Result |
| --- | --- |
| Samples / independent sources | 821 / 92 |
| None / amplitude / phase | 92 / 363 / 366 |
| Square / circular support sources | 73 / 19 |
| Circular class counts | 19 / 75 / 76 |
| Square class counts | 73 / 288 / 290 |
| Shape / zero-based reference pixel | All 256×256 / (128,128) |
| Within-source support changes / maximum PSF pixel difference | 0 / 0 |
| Closest jointly zero-filled dirty/clean/residual pixel to centre | 84.20 pixels |
| Beam minor / major FWHM range | 3.36–4.17 / 4.29–26.06 pixels |
| Radius-80 odd residual energy fraction, class medians | None .4934; amp .2425; phase .7263 |

The parity statistic uses raw, unpaired residuals: `sum(O**2)/sum(R**2)` inside radius 80. It is exploratory training evidence, not classification performance. For a centred point source, amplitude errors tend to produce even structure, while small phase errors tend to produce odd structure. Finite phase angles and CLEAN modify this relation.

Baseline results already published in `ml/runs`: original 14-feature logreg validation balanced accuracy **.6158**, macro F1 **.4964**; four-feature version **.5631**, **.4478**. Compare against both, not just the weaker model. Do not recompute these using new held-out sample inspection during implementation planning.

## Data and preprocessing contract

Reuse `FitsSimulationDataset(..., partition=..., label_criterion="constant_antenna_type")`. Its fixed channel order is `(dirty, clean, residual, psf)`; select/derive CNN inputs in the ML layer without changing that public contract. Labels come from the existing criterion, not the index's nine variant IDs.

Each inference example must be constructible from that sample alone. No paired uncorrupted reference, injected severity, source ID, selected antenna, noise seed, simulated flux, or corruption report becomes an input. Metadata remains available for labels/reporting. Here `clean.fits` means the restored image of the current sample, not the clean class.

Implement one deterministic preprocessing function, accepting the existing four-plane float tensor:

1. Require shape `(4,256,256)`, aligned celestial grids, finite values, and reference pixel `(128,128)` in zero-based coordinates. Confirm reference pixels through the existing validated manifest/FITS paths during preload; the image-only transform does not receive a header. Reject unsupported grids rather than recentering on the brightest corrupted pixel.
2. For v1, infer exterior fill as `(dirty == 0) & (clean == 0) & (residual == 0)`. Verify **no such pixels inside radius 80**, and all required pixels finite. This recognizes this export's shared zero fill; it is not a universal validity-mask reconstruction. A future dataset should store an explicit mask. Never infer support from PSF, which is square even when the images are circular.
3. Set `R = residual[48:209,48:209]`, a 161×161 crop centred exactly at `(80,80)`. Let `r = hypot(x-80,y-80)` on this grid. The odd-sized crop avoids the half-pixel centring error caused by flipping the original even-sized image.
4. Let `A = (32 <= r) & (r < 72)`; compute `s = 1.4826 * median(abs(R[A] - median(R[A])))`. Reject nonfinite/nonpositive `s`; no arbitrary epsilon fallback. This is a robust image-background scale, not a claim of pure thermal noise. Compute it independently per sample; do not subtract another sample or refit from held-out populations.
5. With `J = flip(R, both spatial axes)`, compute `E=(R+J)/2`, `O=(R-J)/2`. Primary input: stack `[R,E,O]`. Control: stack `[R,R,R]`. Divide all channels by **the same residual scale s**. Do not normalize E and O independently, which would remove the measured relative-strength signal. Keep pretrained input filters as supplied; averaging their RGB weights would collapse the linear dependence `R=E+O` and can erase the intended channel distinction.
6. Apply `z = asinh(clip(X/s,-10,10))/asinh(10)` elementwise. Preserve sign. In training, the maximum per-sample fraction of residual pixels beyond ±10s was .095%; record clipping fractions on later splits, without retuning from them.
7. Multiply every channel by the same fixed taper `w(r)`: 1 for `r<=72`, `0.5*(1+cos(pi*(r-72)/8))` for `72<r<80`, and 0 outside. Set zero-weight pixels explicitly to zero before interpolation, so invalid crop corners can never contaminate valid pixels. Map `(z*w+1)/2` to `[0,1]`.
8. Bilinearly resize to `(224,224)` using `align_corners=False`, then normalize by ImageNet mean `(0.485,.456,.406)` and std `(.229,.224,.225)`. Do not invoke the checkpoint's default resize/centre-crop pipeline in addition. Use float tensors throughout.
9. During training only, optionally choose one of the eight square symmetries uniformly (90-degree rotations and reflections), applying it jointly to all channels after preprocessing. Enable this for both initial runs. These transformations preserve even/odd parity about the centre. Validation/test are deterministic. No translation, random crop, colour/brightness jitter, or added noise.

Use a shared 80px aperture, not a different radius per sample. Native minor-beam sampling is already near four pixels; no major-axis beam resampling or beam-angle rotation in v1.

### Geometry tradeoff and admission check

Identical apertures remove the original square/circle edge distinction; they do not erase configuration-dependent noise/PSF structure. Merely adding a mask channel would leave the varying geometry visible to an ordinary CNN.

Radius 80 retains **30.6%** of the full square area. Research computed `DeltaR = R_variant - R_none` **only as a training diagnostic**: `sum((w*DeltaR)**2)/sum(DeltaR**2)` has median **.4376**, 10th–90th percentile **.3241–.5697**, across the 729 corrupted training samples. This is residual-difference energy after imaging, not injected visibility SNR or independent-pixel detection significance. Do not use DeltaR in training/inference.

The crop may reduce low-severity sensitivity substantially. Its advantage is an explicit, simple support control. If a validation/test image fails the fixed support check, report its ID and stop that evaluation; never silently drop it, fill an interior hole, or alter the preprocessing using held-out data. Future full-field or mask-aware work is a separate experiment. Poor cropped-CNN results cannot establish absence of information in the discarded field.

### Channel choices outside the initial experiment

Retain all four products for checks and future work, but feed only the defined derived residual channels. Defer clean (central peak dominates), dirty (ordinary source/PSF dominates), and PSF (source context, not a within-source label signal). If later justified, compare `[R,E,PSF]`; PSF has a positive unit centre in training and must use its own dimensionless normalization. Never divide PSF by residual Jy/beam noise or treat channels as display RGB.

## Model and economical training

Keep the pretrained architecture unchanged except `fc = Linear(512,3)`. Keep the original three-channel stem. Do not add attention, LoRA, auxiliary severity heads, or a custom training framework.

| Stage | Trainable parameters | Optimizer / duration |
| --- | --- | --- |
| Head warmup | `fc` only: 1,539 parameters | AdamW, LR 1e-3, weight decay 1e-4, 5 epochs |
| Limited fine-tune | Convolutions in `layer4[1]` plus `fc`: 4,720,131 parameters | New AdamW; block LR 1e-4, head LR 3e-4, weight decay 1e-4; at most 15 epochs |

Freeze **all** BatchNorm affine parameters and running statistics in both stages. After each `model.train()`, set every BatchNorm module to `eval()`; `requires_grad=False` alone does not freeze its buffers. `layer4[0]`, the stem, and earlier stages stay frozen. No input gradients are needed. Save the best warmup state and start stage two from it; retain the best checkpoint across both stages.

Use batch size 16, FP32, seed 42, and class-weighted cross-entropy with weights `N_train/(3*n_class)`, calculated only from training labels. Shuffle normally, with no weighted sampler layered on top. No label smoothing or scheduler in the first version. Select the checkpoint by validation balanced accuracy, break ties by macro F1 then the earlier epoch, and stop fine-tuning after five epochs without improvement.

Resolve device as CUDA, then MPS, then CPU; expose an override and record the chosen device. The research environment reported arm64 PyTorch 2.14, MPS built but unavailable to that process. Do not promise accelerator availability or runtime. Time one epoch; frozen layers still require forward passes.

Preload and preprocess each requested split once using the existing loader, retaining only processed CPU tensors, labels, and small reporting metadata. This avoids re-reading/decompressing four FITS products every epoch. Float32 inputs occupy about 471 MiB for train and 98 MiB for validation; use workers=0 initially. No persistent cache format is required.

## Minimal ML module changes

- Add `ml/cnn.py`: preprocessing, a small tensor-dataset wrapper, model/freezing setup, train/predict functions, CLI main. Keep preprocessing configuration and input mode in the saved checkpoint.
- Reuse `ml/evaluate.py` with sample-ID-keyed predictions; batches can contain only the labels/metadata it consumes. Preserve existing logreg behaviour.
- Add `ml/cnn_report.py`: use existing metric-table formatting plus `scripts.reporting.QuartoReporter` to generate self-contained `report.html`. The existing logreg writer assumes sklearn models and coefficient plots; do not pass a CNN into it or add a generic reporting framework.
- Add focused `ml/tests/test_cnn.py` checks below. No dataset/imaging/simulation changes.

Proposed command after implementation, from the repository root:

```sh
uv run --project ml python -m ml.cnn \
  --dataset collect/experiments/dataset_v1_20260921_all_psf/dataset.json \
  --input-mode parity --seed 42
```

Second run: replace `parity` with `residual`. Each writes a new timestamped `ml/runs/*_resnet18_<mode>/` directory. Test loading/evaluation requires explicit `--evaluate-test`; leave it off throughout model selection. Fail clearly if weights cannot be downloaded; do not silently train random weights.

## Evaluation and report

Keep the existing 92-source training and 19-source validation partitions, including all available variants. No sample-wise split or augmentation leakage. Only two input modes initially; no cross-validation/hyperparameter sweep. If a candidate exceeds the original logreg's .6158 validation balanced accuracy, repeat the two modes at seeds 43 and 44 before selecting one. Include every seed, not just the best. A small difference on 19 sources is inconclusive; bootstrap uncertainty, if reported, must resample whole source groups with all their variants.

Any improvement belongs to the full representation-plus-classifier experiment. It cannot establish that ImageNet pretraining or a CNN is necessary without a later matched control, such as a simple classifier on even/odd energy features. Do not expand the first two runs into that additional study automatically.

Report overall accuracy, balanced accuracy, macro precision/recall/F1, per-class scores, confusion matrix, and target-SNR breakdowns. At positive SNR, distinguish **detected as non-none** from **correct amp/phase**, both over all corrupted examples at that level. Give clean false-positive rate separately. Existing per-level macros average only classes present at that level; label this clearly so these are not mistaken for three-class macro metrics.

Also report results/counts by original square/circular support and by source; this is an audit field, not a model input. Include two plots: validation/train learning curves and per-SNR detection/type accuracy. An optional small gallery of representative errors is more informative than fictitious CNN coefficient importance.

Artifacts: `checkpoint.pt` (state dict, weights identifier, preprocessing configuration, class map, seed), `predictions.csv` (sample/source, truth, prediction, probabilities, SNR, support group), `history.json`, `report.qmd`, `report.html`, and plot files. Record versions, dataset path/index identity, split counts, class weights, trainable parameter count, optimizer settings, clipping fractions, device, runtime, and the aperture information-loss caveat. Never claim performance from this specification as a result.

## Focused verification

1. Toy centred symmetric/antisymmetric images: E/O isolate the correct component, `R=E+O`, normalization shares one scale, and parity survives joint rotations/reflections. This catches the consequential half-pixel and independent-normalization bugs.
2. Support check: changing pixels only outside the aperture cannot change processed inputs; an interior invalid pixel, nonpositive scale, or unsupported centre fails clearly.
3. One training optimizer step: the head/allowed last-block weights can change; frozen weights and every BatchNorm buffer stay identical. Check predictions are keyed correctly for the existing evaluator.
4. Training-only smoke run on a few sources, then the two specified train/validation runs. Check the generated HTML contains real metrics and plots. Do not add tests of torchvision internals or unrelated CASA operations.

## Sources

- [ResNet-18 checkpoint, size, normalization](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) and [implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html).
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [BatchNorm running statistics](https://docs.pytorch.org/docs/2.12/generated/torch.nn.BatchNorm2d.html).
- [de Jong et al. 2025, radio calibration-image transfer learning](https://academic.oup.com/mnras/article/542/4/3253/8239291), [Raghu et al. 2019, limits of natural-image transfer](https://arxiv.org/abs/1902.07208).
- Local evidence: `scripts/preprocessing/{dataset,fits,partitions,labels}.py`, `scripts/imaging/{fits,metrics}.py`, `docs/dataset_v1_image_support_review.md`, and existing `ml/runs/*/report.md` files. Numerical design choices are proposals informed by the training audit, not prescriptions from those papers.
