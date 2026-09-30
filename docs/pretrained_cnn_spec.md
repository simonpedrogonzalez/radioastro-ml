# Small pretrained CNN experiment for dataset v1

Status: superseded by [the minimal ResNet-18/DINOv2 stack](minimal_nn_approaches_plan.md); the legacy modules were removed. The historical design and results remain below.

Research note: [Pretrained CNN for Constant Antenna Corruption Classification](</Users/u1528314/Documents/Obsidian Vault/Notes/Pretrained CNN for Constant Antenna Corruption Classification.md>).

## Scope and decision

Implement **torchvision ResNet-18 with `ResNet18_Weights.IMAGENET1K_V1`** for `none=0`, `amp=1`, `phase=2`. Use signed residual/symmetric/antisymmetric channels, and one control using the residual repeated three times. This tests whether spatial structure adds information beyond QA scalars. It is specific to the centred-point-source, constant-one-antenna experiment.

**Initial policy: exclude circular-support sources and retain the full square fields.** Apply this as an ML subset filter within the existing partitions. The common-aperture crop is a separate deferred experiment, explicitly not implemented now.

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

These are the earlier audit's measurements across both support groups. The parity statistic uses raw, unpaired residuals: `sum(O**2)/sum(R**2)` inside radius 80; it has not been remeasured for the new full-field/square-only policy. It is exploratory training evidence, not classification performance. For a centred point source, amplitude errors tend to produce even structure, while small phase errors tend to produce odd structure. Finite phase angles and CLEAN modify this relation.

Baseline results already published in `ml/runs`: original 14-feature logreg validation balanced accuracy **.6158**, macro F1 **.4964**; four-feature version **.5631**, **.4478**. These used a different population and are historical context only. During implementation, rerun both baselines on the same retained square-only training/validation samples, preserving their feature definitions and model settings. Compare the CNN against these matched baselines. No further held-out inspection is needed for this documentation update.

## Initial eligibility policy: square support

All FITS arrays are square in shape; eligibility concerns **valid image support**, not array dimensions. For this v1 export, shared exterior zeros in dirty, clean, and residual identify the circular footprint. Never use PSF support or array configuration as the filter.

During preload, form `filled = (dirty == 0) & (clean == 0) & (residual == 0)`. No filled pixels means full-square support; a shared exterior zero region means non-square support and exclusion. Unexpected isolated interior holes or nonfinite data are validation errors, not silently accepted geometry. This is a dataset-specific heuristic, not a universal reconstruction of CASA masks.

Group by source within the requested partition and check that support agrees across all available variants. Exclude **the entire source** if it has circular support; mixed or ambiguous support must fail with a diagnostic. Do not retain selected labels/levels from an excluded source. The existing training audit implies **651 retained samples / 73 sources**, with class counts **73 none, 288 amp, 290 phase**; **170 samples / 19 sources** are excluded. Recompute class weights from the retained labels.

Apply this same fixed rule to validation and, only when explicitly requested, test. Retained validation/test counts are not yet measured. Do not move sources across partitions to replace exclusions or change the policy based on scores. Write excluded source/sample IDs and reasons to the run report; dataset files remain intact. Reject an empty eligible split or one missing a target class before fitting/evaluating. Results apply to square-support sources and do not establish performance on circular ones.

## Data and preprocessing contract

Reuse `FitsSimulationDataset(..., partition=..., label_criterion="constant_antenna_type")`. Its fixed channel order is `(dirty, clean, residual, psf)`; select/derive CNN inputs in the ML layer without changing that public contract. Labels come from the existing criterion, not the index's nine variant IDs.

Each inference example must be constructible from that sample alone. No paired uncorrupted reference, injected severity, source ID, selected antenna, noise seed, simulated flux, or corruption report becomes an input. Metadata remains available for labels/reporting. Here `clean.fits` means the restored image of the current sample, not the clean class.

Implement one deterministic preprocessing function, accepting the existing four-plane float tensor:

1. Require shape `(4,256,256)`, aligned celestial grids, finite values, and reference pixel `(128,128)` in zero-based coordinates. Confirm reference pixels through the existing validated manifest/FITS paths during preload; the image-only transform does not receive a header. Reject unsupported grids rather than recentering on the brightest corrupted pixel.
2. Apply the source eligibility policy above before constructing training tensors. Keep the full square image; **no crop, disk mask, or radial taper** in this experiment.
3. Set `R = residual` on its original 256×256 grid and `r = hypot(x-128,y-128)`. A naive array flip rotates about `(127.5,127.5)`, not the phase centre. Use the actual centre when forming the opposite-pixel comparison below.
4. Let `A = (32 <= r) & (r < 72)`; compute `s = 1.4826 * median(abs(R[A] - median(R[A])))`. Reject nonfinite/nonpositive `s`; no arbitrary epsilon fallback. This is a robust image-background scale, not a claim of pure thermal noise. Compute it independently per sample; do not subtract another sample or refit from held-out populations.
5. On pixels with an in-bounds opposite partner, set `J[y,x]=R[256-y,256-x]`, `E=(R+J)/2`, and `O=(R-J)/2`. Concretely, initialize E/O to zero and fill `[1:,1:]` using `P=R[1:,1:]` and `flip(P, both spatial axes)`. The original first row/column have no partners: leave E/O zero there, while retaining **all** original pixels in R. Do not wrap the unmatched edge to the other side or invent a mirrored measurement. Primary input: `[R,E,O]`; control: `[R,R,R]`. Divide all channels by **the same residual scale s**. Do not normalize E/O independently. Keep pretrained input filters as supplied; averaging their RGB weights can erase the intended channel distinction because `R=E+O` on the paired domain.
6. Apply `z = asinh(clip(X/s,-10,10))/asinh(10)` elementwise. Preserve sign. The earlier maximum .095% clipping fraction was measured only inside radius 80, not over this full-field subset. Audit full-field clipping on retained training samples before training, record the chosen limit in the checkpoint, and never retune it from validation/test distributions.
7. Map `(z+1)/2` to `[0,1]`, with no taper. The annulus in step 4 estimates a scale only; pixels outside it remain model inputs.
8. Bilinearly resize the **entire field** to `(224,224)` using `align_corners=False` and antialiasing, then normalize by ImageNet mean `(0.485,.456,.406)` and std `(.229,.224,.225)`. This changes sampling, not field of view. Do not invoke the checkpoint's default resize/centre-crop pipeline. Use float tensors throughout.
9. During training only, choose one of the eight square symmetries uniformly (90-degree rotations and reflections), applying it jointly to the already-derived channels. Enable this for both initial runs. Do not recompute parity about the transformed tensor's geometric centre: the physical centre is slightly off that centre for this even grid. Validation/test are deterministic. No translation, random crop, colour/brightness jitter, or added noise.

Native minor-beam sampling is already near four pixels; no major-axis beam resampling or beam-angle rotation in v1. The unmatched parity boundary involves 511 of 65,536 pixels (about .78%); R still retains them. This is not the common-aperture crop, and no large-area information-retention claim from that crop applies here.

### Geometry tradeoff

Excluding circular sources removes the mixed-boundary nuisance while keeping square sources' field coverage. It also reduces independent training sources from 92 to 73 and narrows configuration coverage. It does not remove other source/configuration differences. Report exclusions explicitly and do not claim generalization to the excluded support group.

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

Preload, filter, and preprocess each requested split once using the existing loader, retaining only eligible processed CPU tensors, labels, and small reporting metadata. This avoids re-reading/decompressing four FITS products every epoch. Float32 inputs occupy about 374 MiB for the 651 retained training samples; validation memory depends on its eligible count (at most the previous 98 MiB). Use workers=0 initially. No persistent cache format is required.

## Minimal ML module changes

- Add `ml/cnn.py`: source-support filtering, full-field preprocessing, a small tensor-dataset wrapper, model/freezing setup, train/predict functions, CLI main. Record `support_policy="square_only"`, centring/edge handling, preprocessing configuration, and input mode in the saved checkpoint. Do not implement a crop-policy switch yet.
- Reuse `ml/evaluate.py` with sample-ID-keyed predictions; batches can contain only the labels/metadata it consumes. Preserve existing logreg behaviour.
- Add `ml/cnn_report.py`: use existing metric-table formatting plus `scripts.reporting.QuartoReporter` to generate self-contained `report.html`. The existing logreg writer assumes sklearn models and coefficient plots; do not pass a CNN into it or add a generic reporting framework.
- Add focused `ml/tests/test_cnn.py` checks below. No dataset/imaging/simulation changes.

Run from the repository root:

```sh
uv run --project ml python -m ml.cnn \
  --dataset collect/experiments/dataset_v1_20260921_all_psf/dataset.json \
  --input-mode parity --seed 42
```

Second run: replace `parity` with `residual`. Each writes a new timestamped `ml/runs/*_resnet18_<mode>/` directory. Test loading/evaluation requires explicit `--evaluate-test`; leave it off throughout model selection. Fail clearly if weights cannot be downloaded; do not silently train random weights.

Dependencies are managed by `uv sync --project ml` (or automatically by `uv run`). `torchvision` supplies the checkpoint; SciPy checks exterior-connected support masks. The existing CASA environment is unchanged. Optional `--device cpu|mps|cuda`, `--output`, `--warmup-epochs`, and `--finetune-epochs` allow device selection and short smoke runs; scientific defaults remain 5+15 epochs. `--finetune-epochs 0` runs the head only. Each invocation also fits the two matched logreg baselines without altering the standalone logreg defaults.

Verification: `uv run --project ml python -m unittest discover -s ml/tests -v`. Six tests cover existing metrics/evaluation plus support filtering, full-field/parity preprocessing, and frozen weights/BatchNorm. A real training-only smoke run exercised both training stages, checkpoint reload, and the HTML report with two embedded plots; its reused training examples are **not held-out results**.

### Initial implementation runs (seed 42)

Both default train/validation runs completed on MPS. Training retained **651 samples / 73 sources** (73/288/290 by class); validation retained **143 samples / 16 sources** (16/63/64). Excluded: train **170 samples / 19 sources**, validation **27 samples / 3 sources**. All variants of each excluded source were excluded together. **Test was not loaded or evaluated.**

| Model / input | Validation accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| ResNet-18 `[R,E,O]` | .7343 | .8006 | .7018 |
| ResNet-18 `[R,R,R]` | .4545 | .5594 | .4532 |
| Matched 14-feature logreg | .5035 | .5803 | .4906 |
| Matched 4-feature logreg | .4406 | .5330 | .4339 |

Parity selected epoch17 of20; residual-only selected the warmup checkpoint at epoch5 and stopped at epoch10. Mean epoch times were approximately 2.4s and 2.3s (excluding preload/reporting). Maximum full-field training clipping fractions were .0137% (parity) and .0290% (residual-only); the fixed clip limit was unchanged.

Reports, checkpoints, probabilities, support audits, and histories:

- [Parity HTML](../ml/runs/20260922T162116896637Z_resnet18_parity/report.html)
- [Residual-only HTML](../ml/runs/20260922T162316909007Z_resnet18_residual/report.html)

These are local ignored run artifacts. HTML embedding, prediction/exclusion accounting, and source partitions were checked. The parity result is promising, **not a final model-selection conclusion**: seeds43/44 for both modes remain the next experiment before selecting a representation, as specified below. No additional seeds or test evaluation were run during this implementation task.

## Evaluation and report

Keep the existing source assignments, then restrict each partition to eligible square-support sources and include all their available variants. Training retains 73 sources; the retained validation count must be reported after applying the rule. No sample-wise split or augmentation leakage. Only two input modes initially; no cross-validation/hyperparameter sweep. If a candidate exceeds the better of the matched square-only logreg baselines in validation balanced accuracy, repeat the two modes at seeds 43 and 44 before selecting one. Include every seed, not just the best. Small differences on this reduced validation set are inconclusive; bootstrap uncertainty, if reported, must resample whole source groups with all their variants.

Any improvement belongs to the full representation-plus-classifier experiment. It cannot establish that ImageNet pretraining or a CNN is necessary without a later matched control, such as a simple classifier on even/odd energy features. Do not expand the first two runs into that additional study automatically.

Report overall accuracy, balanced accuracy, macro precision/recall/F1, per-class scores, confusion matrix, and target-SNR breakdowns. At positive SNR, distinguish **detected as non-none** from **correct amp/phase**, both over all corrupted examples at that level. Give clean false-positive rate separately. Existing per-level macros average only classes present at that level; label this clearly so these are not mistaken for three-class macro metrics.

Report retained and excluded source/sample counts per partition/class, excluded IDs/reasons, and per-source results on the retained subset. Circular-source classification scores are **not evaluated**, not zero. Include two plots: validation/train learning curves and per-SNR detection/type accuracy. An optional small gallery of representative errors is more informative than fictitious CNN coefficient importance.

Artifacts: `checkpoint.pt` (state dict, weights identifier, preprocessing/support configuration, class map, seed), `predictions.csv` (eligible sample/source, truth, prediction, probabilities, SNR), `exclusions.json` (source/sample IDs and reason), `history.json`, `report.qmd`, `report.html`, and plot files. Record versions, dataset path/index identity, retained/excluded counts, class weights, trainable parameter count, optimizer settings, clipping fractions, device, runtime, and the square-only population limitation. Never claim performance from this specification as a result.

## Focused verification

1. Toy 256×256 images centred at index128: E/O isolate the correct component and `R=E+O` on the paired domain, unmatched E/O edges are neutral, and the full R field is retained. Verify shared normalization and joint augmentation without recomputing parity at a half-pixel centre.
2. Source filtering: all variants of a circular source are excluded, square sources remain in their original partition, and excluded counts/IDs match predictions' eligible population. Mixed support, unexpected interior holes, nonpositive scale, and unsupported centres fail clearly. Do not assert that changing pixels outside radius80 leaves the new full-field input unchanged.
3. One training optimizer step: the head/allowed last-block weights can change; frozen weights and every BatchNorm buffer stay identical. Check predictions are keyed correctly for the existing evaluator.
4. Training-only smoke run on a few sources, then the two specified train/validation runs. Check the generated HTML contains real metrics and plots. Do not add tests of torchvision internals or unrelated CASA operations.

## Deferred experiment — common-aperture crop (not implemented now)

Retain this as a separate future policy, not a preprocessing option or required experiment in the initial implementation. It would include **both square and circular** sources in their original partitions:

1. Require the original aligned 256×256 grids and centre `(128,128)`. The closest jointly zero-filled training pixel was radius84.20; require support everywhere inside the fixed radius80 disk for every input. Never adapt this radius per source or from held-out results.
2. Take `R = residual[48:209,48:209]`, a 161×161 crop centred at `(80,80)`. On this odd grid, `J=flip(R,both spatial axes)` gives the exact centred reflection, so E/O need no unmatched-edge handling.
3. Use the same residual-MAD scale from native annulus `32<=r<72`, channel modes, and signed asinh transform.
4. Apply one shared taper: `w=1` for `r<=72`, `w=0.5*(1+cos(pi*(r-72)/8))` for `72<r<80`, and zero outside. Explicitly zero unsupported corners before interpolation, map `(z*w+1)/2`, resize to224, and apply ImageNet normalization.
5. Report results on the shared square validation subset to compare policies, and separately on circular sources. Overall scores over different populations are not directly comparable. A later input failing support inside the disk must stop that evaluation rather than be silently dropped or assigned a different radius.

The disk retains **30.6%** of the full square area. The earlier training-only diagnostic used `DeltaR = R_variant - R_none`: `sum((w*DeltaR)**2)/sum(DeltaR**2)` had median **.4376**, 10th–90th percentile **.3241–.5697**, across 729 corrupted samples. These are residual-difference energy fractions after imaging, not visibility SNR or classifier performance; DeltaR must never become an inference input. The crop may reduce weak-error sensitivity substantially. Its benefit would be inclusion of more sources with a uniform image boundary. This tradeoff belongs to that future experiment and does not describe the initial full-square policy.

## Sources

- [ResNet-18 checkpoint, size, normalization](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) and [implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html).
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [BatchNorm running statistics](https://docs.pytorch.org/docs/2.12/generated/torch.nn.BatchNorm2d.html).
- [de Jong et al. 2025, radio calibration-image transfer learning](https://academic.oup.com/mnras/article/542/4/3253/8239291), [Raghu et al. 2019, limits of natural-image transfer](https://arxiv.org/abs/1902.07208).
- Local evidence: `scripts/preprocessing/{dataset,fits,partitions,labels}.py`, `scripts/imaging/{fits,metrics}.py`, `docs/dataset_v1_image_support_review.md`, and existing `ml/runs/*/report.md` files. Numerical design choices are proposals informed by the training audit, not prescriptions from those papers.
