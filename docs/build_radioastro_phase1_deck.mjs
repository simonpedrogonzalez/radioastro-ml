import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import {
  createSlideContext,
  ensureArtifactToolWorkspace,
  importArtifactTool,
  saveBlobToFile,
} from "/Users/u1528314/.codex/plugins/cache/openai-primary-runtime/presentations/26.430.10722/skills/presentations/scripts/artifact_tool_utils.mjs";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
const OUTPUT_DIR = path.join(REPO_ROOT, "docs");
const WORKSPACE = path.join("/private/tmp", "codex-presentations", "radioastro-ml-phase1");
const PREVIEW_DIR = path.join(WORKSPACE, "preview");
const LAYOUT_DIR = path.join(WORKSPACE, "layout");
const OUTPUT_PPTX = path.join(OUTPUT_DIR, "radioastro_phase1_10min_deck.pptx");
const CONTACT_SHEET = path.join(OUTPUT_DIR, "radioastro_phase1_10min_contact_sheet.png");
const CONTACT_SHEET_SCRIPT = "/Users/u1528314/.codex/plugins/cache/openai-primary-runtime/presentations/26.430.10722/skills/presentations/scripts/make_contact_sheet.py";
const BUNDLED_PYTHON = "/Users/u1528314/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const SLIDE_SIZE = { width: 1280, height: 720 };

const THEME = {
  bg: "#F4EFE7",
  panel: "#FCFAF5",
  ink: "#132434",
  muted: "#62707A",
  accent: "#0F7A80",
  accent2: "#C8792E",
  accent3: "#DCD1BC",
  accent4: "#E9E2D4",
  dark: "#10202E",
  line: "#D8D0C3",
  ok: "#2E7D60",
  warn: "#B45A3C",
};

const FONTS = {
  title: "Aptos Display",
  body: "Aptos",
  mono: "Aptos Mono",
};

function repoPath(...parts) {
  return path.join(REPO_ROOT, ...parts);
}

function panel(slide, ctx, x, y, w, h, opts = {}) {
  return ctx.addShape(slide, {
    left: x,
    top: y,
    width: w,
    height: h,
    geometry: opts.geometry ?? "rect",
    fill: opts.fill ?? THEME.panel,
    line: opts.line ?? ctx.line(opts.lineColor ?? THEME.line, opts.lineWidth ?? 1),
  });
}

function text(slide, ctx, value, x, y, w, h, opts = {}) {
  const shape = ctx.addText(slide, {
    text: String(value ?? ""),
    left: x,
    top: y,
    width: w,
    height: h,
    fontSize: opts.size ?? 24,
    color: opts.color ?? THEME.ink,
    bold: Boolean(opts.bold),
    typeface: opts.face ?? FONTS.body,
    align: opts.align ?? "left",
    valign: opts.valign ?? "top",
    fill: opts.fill ?? "#00000000",
    line: opts.line ?? ctx.line(),
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  });
  if (opts.lineSpacing) {
    shape.text.lineSpacing = opts.lineSpacing;
  }
  return shape;
}

function addKicker(slide, ctx, label) {
  panel(slide, ctx, 64, 48, 110, 6, { fill: THEME.accent2, lineWidth: 0 });
  text(slide, ctx, label, 64, 62, 360, 20, {
    size: 13,
    bold: true,
    face: FONTS.mono,
    color: THEME.accent,
  });
}

function addTitle(slide, ctx, kicker, titleText, subtitle) {
  addKicker(slide, ctx, kicker);
  text(slide, ctx, titleText, 64, 86, 760, 74, {
    size: 37,
    bold: true,
    face: FONTS.title,
    lineSpacing: 1.05,
  });
  if (subtitle) {
    text(slide, ctx, subtitle, 64, 162, 760, 56, {
      size: 18,
      color: THEME.muted,
      lineSpacing: 1.18,
    });
  }
}

function addFooter(slide, ctx, slideNumber) {
  panel(slide, ctx, 64, 686, 1152, 1, { fill: THEME.line, lineWidth: 0 });
  text(slide, ctx, "radioastro-ml | Phase I infrastructure deck", 64, 694, 620, 16, {
    size: 12,
    color: THEME.muted,
    face: FONTS.mono,
  });
  text(slide, ctx, String(slideNumber).padStart(2, "0"), 1176, 692, 40, 16, {
    size: 12,
    color: THEME.muted,
    face: FONTS.mono,
    align: "right",
    bold: true,
  });
}

function bulletBlock(items) {
  return items.map((item) => `- ${item}`).join("\n\n");
}

function chip(slide, ctx, label, x, y, w, opts = {}) {
  panel(slide, ctx, x, y, w, 34, {
    fill: opts.fill ?? THEME.accent4,
    lineColor: opts.lineColor ?? THEME.line,
    lineWidth: 1,
  });
  text(slide, ctx, label, x + 12, y + 8, w - 24, 18, {
    size: 13,
    bold: true,
    face: FONTS.mono,
    color: opts.color ?? THEME.ink,
    align: "center",
  });
}

async function optimizedImage(assetDir, imagePath, maxPx = 1600) {
  if (!assetDir) return imagePath;
  await fs.mkdir(assetDir, { recursive: true });
  const parsed = path.parse(imagePath);
  const safeBase = path.relative(REPO_ROOT, imagePath).replaceAll(path.sep, "__");
  const outputPath = path.join(assetDir, `${safeBase.replace(parsed.ext, "")}__${maxPx}${parsed.ext}`);
  try {
    await fs.access(outputPath);
    return outputPath;
  } catch {}

  const resizeCode = [
    "from PIL import Image",
    "import sys",
    "src, dst, max_px = sys.argv[1], sys.argv[2], int(sys.argv[3])",
    "img = Image.open(src)",
    "img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)",
    "save_kwargs = {'optimize': True}",
    "if dst.lower().endswith(('.jpg', '.jpeg')) and img.mode in ('RGBA', 'LA', 'P'):",
    "    img = img.convert('RGB')",
    "if dst.lower().endswith('.jpg') or dst.lower().endswith('.jpeg'):",
    "    save_kwargs['quality'] = 88",
    "img.save(dst, **save_kwargs)",
  ].join("\n");
  const result = spawnSync(BUNDLED_PYTHON, ["-c", resizeCode, imagePath, outputPath, String(maxPx)], {
    encoding: "utf8",
  });
  if (result.status !== 0) {
    return imagePath;
  }
  return outputPath;
}

async function framedImage(slide, ctx, imagePath, x, y, w, h, caption, opts = {}) {
  panel(slide, ctx, x, y, w, h, {
    fill: opts.panelFill ?? THEME.panel,
    lineColor: opts.lineColor ?? THEME.line,
    lineWidth: 1,
  });
  const captionHeight = caption ? 44 : 0;
  const stagedPath = await optimizedImage(ctx.assetDir, imagePath, opts.maxPx ?? 1600);
  await ctx.addImage(slide, {
    path: stagedPath,
    left: x + 10,
    top: y + 10,
    width: w - 20,
    height: h - 20 - captionHeight,
    fit: opts.fit ?? "contain",
  });
  if (caption) {
    text(slide, ctx, caption, x + 12, y + h - 30, w - 24, 24, {
      size: 12,
      color: opts.captionColor ?? THEME.muted,
      lineSpacing: 1.0,
    });
  }
}

function stepBox(slide, ctx, number, titleText, bodyText, x) {
  panel(slide, ctx, x, 236, 180, 192, { fill: THEME.panel });
  panel(slide, ctx, x + 18, 252, 38, 38, {
    fill: THEME.accent,
    lineColor: THEME.accent,
    lineWidth: 0,
  });
  text(slide, ctx, String(number), x + 18, 262, 38, 18, {
    size: 16,
    bold: true,
    face: FONTS.mono,
    color: "#FFFFFF",
    align: "center",
  });
  text(slide, ctx, titleText, x + 18, 308, 144, 40, {
    size: 20,
    bold: true,
    face: FONTS.title,
    lineSpacing: 1.08,
  });
  text(slide, ctx, bodyText, x + 18, 354, 144, 54, {
    size: 13,
    color: THEME.muted,
    lineSpacing: 1.12,
  });
}

function questionCard(slide, ctx, label, bodyText, x, y, w = 360, h = 104) {
  panel(slide, ctx, x, y, w, h, { fill: THEME.panel });
  text(slide, ctx, label, x + 16, y + 14, w - 32, 18, {
    size: 13,
    bold: true,
    face: FONTS.mono,
    color: THEME.accent,
  });
  text(slide, ctx, bodyText, x + 16, y + 38, w - 32, h - 46, {
    size: 18,
    face: FONTS.body,
    lineSpacing: 1.15,
  });
}

async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  panel(slide, ctx, 786, 0, 494, 720, { fill: THEME.dark, lineWidth: 0 });
  addKicker(slide, ctx, "COSMICAI / RADIOASTRO-ML");
  text(slide, ctx, "Accelerating Radio Interferometry Error Detection with AI", 64, 98, 640, 128, {
    size: 52,
    bold: true,
    face: FONTS.title,
    lineSpacing: 1.04,
  });
  text(
    slide,
    ctx,
    "Main claim: this is still a Phase I infrastructure effort. The current win is a repeatable path from real VLA calibrator observations to controlled faults, standardized imaging, and future labels.",
    64,
    248,
    620,
    104,
    {
      size: 20,
      color: THEME.muted,
      lineSpacing: 1.18,
    },
  );

  chip(slide, ctx, "real public VLA data", 64, 394, 180);
  chip(slide, ctx, "controlled corruptions", 256, 394, 196);
  chip(slide, ctx, "diagnostics before ML", 464, 394, 208);

  questionCard(
    slide,
    ctx,
    "THESIS",
    "The project should not claim an ML result yet. It should claim that the data, corruption, imaging, and QA machinery is finally becoming trustworthy enough to support one.",
    64,
    454,
    620,
    146,
  );

  await framedImage(
    slide,
    ctx,
    repoPath("images", "sample_diagnosis", "all_samples_contact_sheet.png"),
    822,
    92,
    422,
    520,
    "Extracted-image contact sheet from the current sample-building workflow.",
    {
      panelFill: "#F9F5EC",
      fit: "contain",
      captionColor: "#C6D1DA",
    },
  );

  text(slide, ctx, "Pipeline focus: data -> corruption -> CASA imaging -> diagnostics -> future labels", 822, 628, 422, 36, {
    size: 15,
    color: "#E3E7EB",
    lineSpacing: 1.08,
  });

  addFooter(slide, ctx, 1);
}

async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  addTitle(
    slide,
    ctx,
    "PROBLEM",
    "Why this is hard: image artifacts do not come with clean labels",
    "Calibration issues, imaging choices, and bad visibilities can all create structure that looks deceptively real.",
  );

  text(slide, ctx, bulletBlock([
    "Expert inspection still matters, but it does not scale.",
    "Naturally bad images are not enough if the cause of the failure is ambiguous.",
    "Any future ML effort needs examples where the underlying cause is actually known.",
  ]), 64, 244, 420, 220, {
    size: 22,
    lineSpacing: 1.18,
  });

  questionCard(
    slide,
    ctx,
    "WHY REAL DATA MATTERS",
    "The project is trying to keep the sky and uv-coverage realistic while controlling the fault itself.",
    64,
    498,
    420,
    126,
  );

  chip(slide, ctx, "calibration", 528, 214, 120, { fill: "#E7F1F2", color: THEME.accent });
  chip(slide, ctx, "imaging", 660, 214, 100, { fill: "#F6EBDD", color: THEME.accent2 });
  chip(slide, ctx, "bad data", 772, 214, 110, { fill: "#EFE8E0", color: THEME.warn });

  await framedImage(
    slide,
    ctx,
    repoPath("images", "sample_diagnosis", "0259+077", "0259+077_clean_comparison.png"),
    528,
    256,
    320,
    182,
    "Example: persistent spokes that line up with the PSF.",
  );
  await framedImage(
    slide,
    ctx,
    repoPath("images", "sample_diagnosis", "0739+016", "0739+016_clean_comparison.png"),
    876,
    256,
    320,
    182,
    "Example: the dirty image already contains suspicious structure.",
  );
  await framedImage(
    slide,
    ctx,
    repoPath("images", "sample_diagnosis", "0739+016", "baseline_uv.png"),
    528,
    468,
    668,
    170,
    "Poor or unusual uv coverage can be part of the explanation, which is exactly why the labels are tricky.",
  );

  addFooter(slide, ctx, 2);
}

async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  addTitle(
    slide,
    ctx,
    "APPROACH",
    "Core workflow: turn uncertain bad images into examples with known causes",
    "The project uses real calibrator observations, then adds controlled faults so the diagnostic outputs can be interpreted against a known intervention.",
  );

  const xs = [80, 306, 532, 758, 984];
  const steps = [
    ["Public calibrator data", "Search public VLA calibrator observations and select usable projects."],
    ["Extraction", "Keep the calibrator data and build a consistent sample-processing path."],
    ["Controlled corruption", "Inject gain-table based phase or amplitude faults with known settings."],
    ["CASA imaging", "Re-image the data with a more standardized grid and recipe."],
    ["Diagnostics / labels", "Compare visibilities, gains, images, PSF, residuals, and uv coverage."],
  ];

  steps.forEach((entry, index) => {
    stepBox(slide, ctx, index + 1, entry[0], entry[1], xs[index]);
    if (index < xs.length - 1) {
      panel(slide, ctx, xs[index] + 180, 320, 40, 8, { fill: THEME.accent3, lineWidth: 0 });
    }
  });

  questionCard(
    slide,
    ctx,
    "WHAT GETS COMPARED",
    "clean vs corrupted visibilities | gain tables | dirty image | clean image | residual image | PSF | uv coverage",
    120,
    500,
    420,
    94,
  );
  questionCard(
    slide,
    ctx,
    "WHY THIS MATTERS",
    "The labels come from the intervention itself, not from guessing after the fact which artifact 'looks like' a certain problem.",
    648,
    500,
    512,
    94,
  );

  addFooter(slide, ctx, 3);
}

async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  addTitle(
    slide,
    ctx,
    "STATUS",
    "Current scope: still Phase I, and that is the honest framing",
    "The most important progress has been making the pre-ML workflow reproducible enough to trust.",
  );

  text(slide, ctx, "Three-phase roadmap", 64, 232, 420, 26, {
    size: 16,
    bold: true,
    face: FONTS.mono,
    color: THEME.accent,
  });

  const phaseY = 278;
  const phaseW = 214;
  const gap = 24;
  const phaseXs = [64, 64 + phaseW + gap, 64 + (phaseW + gap) * 2];
  const phaseData = [
    ["Phase I", "data + corruption + diagnostics", THEME.accent, "#FFFFFF"],
    ["Phase II", "pilot labeled dataset", THEME.accent4, THEME.ink],
    ["Phase III", "ML experiments", THEME.accent4, THEME.ink],
  ];
  phaseData.forEach((item, index) => {
    panel(slide, ctx, phaseXs[index], phaseY, phaseW, 182, {
      fill: item[2],
      lineColor: item[2],
      lineWidth: 0,
    });
    text(slide, ctx, item[0], phaseXs[index] + 20, phaseY + 24, phaseW - 40, 40, {
      size: 28,
      bold: true,
      face: FONTS.title,
      color: item[3],
    });
    text(slide, ctx, item[1], phaseXs[index] + 20, phaseY + 82, phaseW - 40, 64, {
      size: 20,
      color: item[3],
      lineSpacing: 1.12,
    });
  });

  questionCard(
    slide,
    ctx,
    "THIS DECK IS",
    bulletBlock([
      "data pipeline progress",
      "corruption validation",
      "diagnostic standardization",
    ]),
    810,
    252,
    360,
    122,
  );
  questionCard(
    slide,
    ctx,
    "THIS DECK IS NOT",
    bulletBlock([
      "a large labeled dataset",
      "a benchmarked classifier",
      "a final ML result",
    ]),
    810,
    400,
    360,
    122,
  );

  questionCard(
    slide,
    ctx,
    "WHY THE PHASE I LABEL MATTERS",
    "It prevents overselling the work and keeps the next milestone concrete: produce a small, clean pilot dataset before claiming ML success.",
    64,
    516,
    1144,
    98,
  );

  addFooter(slide, ctx, 4);
}

async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  addTitle(
    slide,
    ctx,
    "DATA + TOOLS",
    "Data acquisition is now repeatable, even if the dataset is not fully curated yet",
    "The README shows steady progress from initial sample batches toward a working path for selecting, downloading, extracting, and imaging candidate calibrators.",
  );

  await framedImage(
    slide,
    ctx,
    repoPath("images", "better_imaging", "new_set.png"),
    64,
    232,
    684,
    352,
    "Larger sample batch after the data-requesting and download pipeline was automated further.",
  );

  text(slide, ctx, bulletBlock([
    "Search public VLA calibrator observations and prioritize useful projects.",
    "Download only the calibrator-relevant data products.",
    "Extract and image candidate samples with a consistent pipeline.",
    "Keep vetting because some projects are too flagged or otherwise unsuitable.",
  ]), 786, 236, 420, 186, {
    size: 20,
    lineSpacing: 1.18,
  });

  questionCard(
    slide,
    ctx,
    "BY-PRODUCT 1",
    "A structured VLA calibrator catalog makes selection more programmable and tracks metadata such as band, flux, and uv-limit information.",
    786,
    448,
    420,
    94,
  );
  questionCard(
    slide,
    ctx,
    "BY-PRODUCT 2",
    "The NRAO archive querying / downloading utilities were split into their own fetcher workflow to make requests and manifests more reproducible.",
    786,
    554,
    420,
    94,
  );

  addFooter(slide, ctx, 5);
}

async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  addTitle(
    slide,
    ctx,
    "CORRUPTION",
    "Controlled fault injection is shifting from simulator exploration to gain-table based experiments",
    "The target is not just to make corrupted images. The target is to make corrupted images whose cause is controlled and interpretable.",
  );

  text(slide, ctx, bulletBlock([
    "Initial focus is on antenna-based phase and amplitude gain errors.",
    "Time-varying patterns such as sine-like drift and fBM-style drift have been explored.",
    "README notes that some simulator modes were useful for understanding the problem, but not ideal for reliable dataset generation.",
  ]), 64, 240, 360, 242, {
    size: 20,
    lineSpacing: 1.18,
  });

  questionCard(
    slide,
    ctx,
    "TAKEAWAY",
    "Move toward corruption settings that can be reasoned about physically and reproduced cleanly from run to run.",
    64,
    516,
    360,
    108,
  );

  await framedImage(
    slide,
    ctx,
    repoPath("images", "mycorr", "sine_int_corrtab.png"),
    470,
    246,
    228,
    324,
    "Sinusoidal corruption stored in the gain table.",
  );
  await framedImage(
    slide,
    ctx,
    repoPath("images", "mycorr", "fbm.png"),
    722,
    246,
    228,
    324,
    "fBM-style drift as a more stochastic time-varying pattern.",
  );
  await framedImage(
    slide,
    ctx,
    repoPath("images", "sine_corruption.png"),
    974,
    246,
    234,
    324,
    "Image-domain effect after a controlled corruption is applied.",
  );

  addFooter(slide, ctx, 6);
}

async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  addTitle(
    slide,
    ctx,
    "VALIDATION",
    "The labels only matter if the injected faults behave the way the calibration model predicts",
    "Closure tests are a sanity check on what kind of error is actually being injected, and what gain calibration should or should not be able to remove.",
  );

  text(slide, ctx, bulletBlock([
    "Antenna-based phase corruption preserves closure phase.",
    "Baseline-only corruption violates closure.",
    "Baseline-only corruption is not removed by antenna-based gaincal, which separates it from per-antenna faults.",
  ]), 64, 246, 320, 236, {
    size: 20,
    lineSpacing: 1.18,
  });

  questionCard(
    slide,
    ctx,
    "WHY THIS IS IMPORTANT",
    "This is the difference between a meaningful label and a visually interesting artifact with unclear physical interpretation.",
    64,
    516,
    320,
    112,
  );

  await framedImage(
    slide,
    ctx,
    repoPath("images", "closure", "constant_per_antenna", "closure_phase_vs_time.png"),
    428,
    246,
    250,
    328,
    "Per-antenna corruption: closure stays consistent.",
  );
  await framedImage(
    slide,
    ctx,
    repoPath("images", "closure", "one_baseline", "closure_phase_vs_time.png"),
    704,
    246,
    250,
    328,
    "One-baseline corruption: closure breaks.",
  );
  await framedImage(
    slide,
    ctx,
    repoPath("images", "recovery_zoomed", "zoom_gtab_corrupt_recovered.png"),
    980,
    246,
    228,
    328,
    "Recovery behavior after gaincal is another check on error class.",
  );

  addFooter(slide, ctx, 7);
}

async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  addTitle(
    slide,
    ctx,
    "IMAGING",
    "The imaging and diagnostics pipeline is more standardized, but the results are still mixed",
    "README progress here is about consistency and interpretation, not a magic correction recipe.",
  );

  text(slide, ctx, bulletBlock([
    "Outputs now include dirty image, clean image, residual image, PSF, uv coverage, and summary metrics.",
    "Beam-based cell size and field-of-view choices reduce arbitrary imaging settings.",
    "Several fixes were tested: uv-lim recalibration, more MT-MFS terms / iterations, box masks, and the VLA selfcal pipeline.",
  ]), 64, 238, 372, 246, {
    size: 19,
    lineSpacing: 1.16,
  });

  questionCard(
    slide,
    ctx,
    "CAUTION",
    "Lower residual sigma or nicer scalar metrics do not automatically mean the image is scientifically better. README explicitly warns that CLEAN can improve metrics while still fitting artifacts.",
    64,
    516,
    372,
    132,
  );

  await framedImage(
    slide,
    ctx,
    repoPath("images", "better_imaging", "old_beam_size.png"),
    480,
    232,
    342,
    180,
    "Old beam-size estimate.",
  );
  await framedImage(
    slide,
    ctx,
    repoPath("images", "better_imaging", "new_beam_size.png"),
    850,
    232,
    342,
    180,
    "Beam-based first-pass estimate.",
  );
  await framedImage(
    slide,
    ctx,
    repoPath("images", "better_imaging", "beam_size_issue_before.png"),
    480,
    432,
    342,
    180,
    "Before: problematic field-of-view / grid choice.",
  );
  await framedImage(
    slide,
    ctx,
    repoPath("images", "better_imaging", "beam_size_issue_after.png"),
    850,
    432,
    342,
    180,
    "After: more reasonable imaging setup.",
  );

  addFooter(slide, ctx, 8);
}

async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  addTitle(
    slide,
    ctx,
    "VISIBILITY QA",
    "New work in progress: robust anomaly scoring directly in the visibility domain",
    "This adds a bad-data QA pass that can point to suspicious antennas, baselines, time bins, or channel ranges before relying only on image artifacts.",
  );

  await framedImage(
    slide,
    ctx,
    repoPath("images", "badant", "bad_uv_dist_vs_amp_contact_sheet_20260504T113900.png"),
    64,
    236,
    728,
    374,
    "Visibility-QA contact sheet for the BAD_UV_DIST_VS_AMP batch.",
  );

  questionCard(
    slide,
    ctx,
    "PIPELINE",
    "1. take log-amplitude\n2. bin by uv-distance / channel / SPW\n3. compute robust local z-scores\n4. aggregate strong outliers by antenna, baseline, time, and channel groups",
    836,
    238,
    372,
    188,
  );
  questionCard(
    slide,
    ctx,
    "THRESHOLDS IN README",
    "strong outlier: |z| >= 8\nmoderate outlier: |z| >= 5\ncandidate groups are scored by coverage, enrichment, bad fraction, and data loss",
    836,
    448,
    372,
    148,
  );
  chip(slide, ctx, "diagnostic tool, not automatic flagging", 836, 614, 322, {
    fill: "#E8F2F3",
    color: THEME.accent,
    lineColor: "#B5D0D2",
  });

  addFooter(slide, ctx, 9);
}

async function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  addTitle(
    slide,
    ctx,
    "PROGRESS",
    "What exists now, and what still blocks a serious ML claim",
    "The right summary is not 'ML works.' The right summary is 'the pilot-data-generation workflow is finally taking shape.'",
  );

  panel(slide, ctx, 64, 236, 544, 356, { fill: "#F2F8F5", lineColor: "#C7DBD2" });
  panel(slide, ctx, 672, 236, 544, 356, { fill: "#FBF1ED", lineColor: "#E2C9BF" });

  text(slide, ctx, "What exists now", 88, 260, 240, 30, {
    size: 24,
    bold: true,
    face: FONTS.title,
    color: THEME.ok,
  });
  text(slide, ctx, bulletBlock([
    "candidate-data workflow",
    "structured calibrator catalog",
    "archive querying / download tooling",
    "gain-table corruption experiments",
    "closure / recovery validation",
    "automated imaging diagnostics",
    "visibility anomaly prototype",
  ]), 88, 308, 480, 254, {
    size: 19,
    lineSpacing: 1.16,
  });

  text(slide, ctx, "What still blocks ML", 696, 260, 260, 30, {
    size: 24,
    bold: true,
    face: FONTS.title,
    color: THEME.warn,
  });
  text(slide, ctx, bulletBlock([
    "final fault-class list",
    "final severity ranges",
    "sample vetting / rejection rules",
    "pilot labeled dataset",
    "first ML benchmark",
    "generalization checks across calibrators / setups",
  ]), 696, 308, 480, 224, {
    size: 19,
    lineSpacing: 1.16,
  });

  questionCard(
    slide,
    ctx,
    "NEXT MILESTONE",
    "Build a small, clean pilot dataset first. Only then is a baseline classifier worth treating as evidence rather than as a curiosity.",
    64,
    612,
    1152,
    62,
  );

  addFooter(slide, ctx, 10);
}

async function slide11(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = THEME.bg;

  addTitle(
    slide,
    ctx,
    "NEXT STEP",
    "Near-term plan: produce a small pilot dataset and ask the right questions now",
    "The README's own framing is useful here: the next milestone is a clean pilot, not a full-scale ML system.",
  );

  const planX = 64;
  const planW = 454;
  const planY = [230, 308, 386, 464, 542];
  const planLabels = [
    "Choose the first fault classes.",
    "Freeze one imaging / diagnostics recipe.",
    "Define sample rejection rules.",
    "Generate a pilot labeled dataset.",
    "Run a small sanity-check baseline classifier.",
  ];

  planY.forEach((y, index) => {
    panel(slide, ctx, planX, y, planW, 56, {
      fill: index === 0 ? "#E8F2F3" : THEME.panel,
      lineColor: index === 0 ? "#B6D0D2" : THEME.line,
    });
    text(slide, ctx, String(index + 1), planX + 18, y + 18, 24, 16, {
      size: 14,
      face: FONTS.mono,
      bold: true,
      color: THEME.accent,
    });
    text(slide, ctx, planLabels[index], planX + 54, y + 16, planW - 72, 24, {
      size: 20,
      lineSpacing: 1.08,
    });
    if (index < planY.length - 1) {
      panel(slide, ctx, planX + 26, y + 56, 4, 22, { fill: THEME.accent3, lineWidth: 0 });
    }
  });

  text(slide, ctx, "Feedback needed", 596, 236, 260, 28, {
    size: 26,
    bold: true,
    face: FONTS.title,
  });
  questionCard(slide, ctx, "QUESTION 1", "Which fault classes should come first?", 596, 282, 288, 140);
  questionCard(slide, ctx, "QUESTION 2", "What should count as a usable calibrator sample?", 596, 430, 288, 140);
  questionCard(slide, ctx, "QUESTION 3", "Which products should be ML inputs and what is the first success criterion?", 922, 282, 288, 140);
  questionCard(slide, ctx, "QUESTION 4", "What output is most useful in the near term: pilot dataset, diagnostic pipeline, software tools, or first ML baseline?", 922, 430, 288, 140);

  addFooter(slide, ctx, 11);
}

async function buildDeck() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });
  await ensureArtifactToolWorkspace(WORKSPACE);
  const artifact = await importArtifactTool(WORKSPACE);
  const { Presentation, PresentationFile } = artifact;
  const presentation = Presentation.create({ slideSize: SLIDE_SIZE });
  const ctx = createSlideContext(artifact, {
    slideSize: SLIDE_SIZE,
    outputDir: OUTPUT_DIR,
    assetDir: path.join(WORKSPACE, "assets"),
    workspaceDir: WORKSPACE,
    titleFont: FONTS.title,
    bodyFont: FONTS.body,
    monoFont: FONTS.mono,
  });

  const builders = [
    slide01,
    slide02,
    slide03,
    slide04,
    slide05,
    slide06,
    slide07,
    slide08,
    slide09,
    slide10,
    slide11,
  ];

  for (const builder of builders) {
    await builder(presentation, ctx);
  }

  const previewPaths = [];
  for (let index = 0; index < builders.length; index += 1) {
    const slide = presentation.slides.getItem(index);
    const previewPath = path.join(PREVIEW_DIR, `slide-${String(index + 1).padStart(2, "0")}.png`);
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await saveBlobToFile(png, previewPath);
    previewPaths.push(previewPath);

    const layoutPath = path.join(LAYOUT_DIR, `slide-${String(index + 1).padStart(2, "0")}.layout.json`);
    const layout = await presentation.export({ slide, format: "layout" });
    await fs.writeFile(layoutPath, await layout.text(), "utf8");
  }

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUTPUT_PPTX);

  const result = spawnSync(BUNDLED_PYTHON, [CONTACT_SHEET_SCRIPT, "--output", CONTACT_SHEET, ...previewPaths], {
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(
      [
        "Contact sheet generation failed.",
        result.stdout.trim(),
        result.stderr.trim(),
      ].filter(Boolean).join("\n"),
    );
  }

  const summary = {
    outputPptx: OUTPUT_PPTX,
    contactSheet: CONTACT_SHEET,
    previewDir: PREVIEW_DIR,
    layoutDir: LAYOUT_DIR,
    slideCount: builders.length,
  };
  console.log(JSON.stringify(summary, null, 2));
}

buildDeck().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
