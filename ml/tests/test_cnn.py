import csv
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from ml import dinov2, resnet18
from ml.compare_nn_runs import combine_runs
from ml.nn_common import (IMAGENET_MEAN, IMAGENET_STD, LoadedSplit, PARITY_CHANNELS,
                          RESIDUAL_RRR_CHANNELS, PreparedSplit, load_data,
                          parity_channels, predict, prepare, residual_rrr_input,
                          rotate, train_job)
from ml.run_nn_experiments import _complete, experiments, jobs, write_report
from ml.task_evaluation import evaluate_task


def loaded(train_shift=0.0, validation_shift=100.0):
    grid = torch.linspace(-1, 1, 224 * 224).reshape(224, 224)

    def split(shift):
        raw = torch.stack([torch.stack([grid + shift + sample + channel
                                       for channel in range(4)]) for sample in range(3)])
        parity = raw[:, :2].clone()
        return LoadedSplit(raw, parity, torch.tensor([0, 1, 2]),
                           ("a", "b", "c"), ("s1", "s2", "s3"),
                           ({"corruption_snr_target": 0, "sample_kind": "baseline"},
                            {"corruption_snr_target": 10, "sample_kind": "gain"},
                            {"corruption_snr_target": 10, "sample_kind": "gain"}))
    controls = LoadedSplit(torch.stack([grid.repeat(4, 1, 1) + validation_shift]),
                           torch.stack([grid.repeat(2, 1, 1) + validation_shift]),
                           torch.tensor([0]), ("noise",), ("s1",),
                           ({"corruption_snr_target": 10,
                             "sample_kind": "increased_noise"},))
    return {"train": split(train_shift), "validation": split(validation_shift),
            "noise_controls": controls}


class PreprocessingTests(unittest.TestCase):
    def test_dataset_loading_and_resize(self):
        partitions = []
        samples = [{"image": torch.randn(4, 256, 256), "label": label,
                    "sample_id": f"sample-{label}",
                    "label_metadata": {"corruption_snr_target": 0 if label == 0 else 10,
                                       "sample_kind": "baseline" if label == 0 else "gain"}}
                   for label in range(3)]
        controls = [{"image": torch.randn(4, 256, 256), "label": 0,
                     "sample_id": "noise", "label_metadata": {
                         "corruption_snr_target": 10, "sample_kind": "increased_noise"}}]
        def dataset(*args, **kwargs):
            partitions.append(kwargs["partition"])
            return controls if kwargs["index"] == "dataset_noise_controls.json" else samples
        with tempfile.TemporaryDirectory() as temporary, \
             patch("ml.nn_common.FitsSimulationDataset", side_effect=dataset), \
             patch("ml.nn_common.source_dataset_id", return_value="source"):
            root = Path(temporary); (root / "dataset.json").touch()
            (root / "dataset_noise_controls.json").touch()
            splits, labels = load_data(root / "dataset.json")
        self.assertEqual(labels, (0, 1, 2))
        self.assertEqual(splits["train"].raw.shape, (3, 4, 224, 224))
        self.assertEqual(splits["validation"].parity.shape, (3, 2, 224, 224))
        self.assertEqual(splits["noise_controls"].labels.tolist(), [0])
        self.assertEqual(partitions, ["train", "val", "val"])

    def test_parity_geometry(self):
        y, x = torch.meshgrid(torch.arange(256) - 128, torch.arange(256) - 128,
                              indexing="ij")
        even, odd = (x.square() + y.square()).float(), (x + y).float()
        result = parity_channels(even + odd)
        torch.testing.assert_close(result[0, 1:, 1:], even[1:, 1:])
        torch.testing.assert_close(result[1, 1:, 1:], odd[1:, 1:])
        self.assertEqual(result[:, 0].count_nonzero(), 0)
        self.assertEqual(result[:, :, 0].count_nonzero(), 0)

    def test_residual_rrr_uses_per_image_mad_asinh_and_imagenet_normalization(self):
        generator = torch.Generator().manual_seed(12)
        residual = torch.randn((256, 256), generator=generator)
        image, scale = residual_rrr_input(residual)
        y, x = torch.meshgrid(torch.arange(256, dtype=residual.dtype),
                              torch.arange(256, dtype=residual.dtype), indexing="ij")
        annulus = residual[(torch.hypot(x - 128, y - 128) >= 32) &
                           (torch.hypot(x - 128, y - 128) < 72)]
        expected_scale = 1.4826 * (annulus - annulus.median()).abs().median()
        self.assertAlmostEqual(scale, float(expected_scale), places=6)
        self.assertEqual(image.shape, (3, 224, 224))
        restored = image * torch.tensor(IMAGENET_STD)[:, None, None]
        restored += torch.tensor(IMAGENET_MEAN)[:, None, None]
        torch.testing.assert_close(restored[0], restored[1])
        torch.testing.assert_close(restored[1], restored[2])
        self.assertGreaterEqual(float(restored.min()), 0)
        self.assertLessEqual(float(restored.max()), 1)
        with self.assertRaisesRegex(ValueError, "MAD scale"):
            residual_rrr_input(torch.ones(256, 256))

    def test_training_only_normalization_and_zero_slots(self):
        splits = loaded()
        prepared, settings = prepare(splits, ("dirty", "residual"), (0, 1, 2))
        self.assertAlmostEqual(float(prepared["train"].images[:, 0].mean()), 0, places=5)
        self.assertGreater(float(prepared["validation"].images[:, 0].mean()), 20)
        self.assertEqual(prepared["train"].images[:, 1].count_nonzero(), 0)
        self.assertEqual(prepared["train"].images[:, 3].count_nonzero(), 0)
        self.assertEqual(settings["included"], ["dirty", "residual"])
        parity, parity_settings = prepare(splits, PARITY_CHANNELS, (0, 1, 2))
        self.assertEqual(parity["train"].images.shape[1], 2)
        self.assertEqual(parity_settings["source"], "parity")

        rrr = {name: LoadedSplit(split.raw, split.parity, split.labels, split.sample_ids,
                                split.source_ids, split.metadata,
                                torch.randn(len(split.labels), 3, 224, 224))
               for name, split in splits.items()}
        prepared_rrr, settings = prepare(rrr, RESIDUAL_RRR_CHANNELS, (0, 1, 2))
        self.assertEqual(prepared_rrr["train"].images.shape[1], 3)
        self.assertEqual(settings["source"], "residual_rrr")
        self.assertEqual(settings["transform"]["kind"], "signed_asinh")

    def test_rotation_is_joint(self):
        image = torch.stack([torch.arange(16).reshape(4, 4) + 100 * channel
                             for channel in range(4)])
        transformed = rotate(image, 3)
        for channel in range(4):
            torch.testing.assert_close(transformed[channel],
                                       torch.rot90(image[channel], 3, (-2, -1)))


class ResNetTests(unittest.TestCase):
    def test_trainable_policies_and_frozen_batchnorm(self):
        expected = {
            "head": lambda name: name.startswith("fc."),
            "last": lambda name: name.startswith("fc.") or name.startswith("layer4.1."),
            "all": lambda name: True,
        }
        for mode, allowed in expected.items():
            model, groups, set_mode = resnet18.build(3, 4, mode, weights=None)
            self.assertTrue(all(parameter.requires_grad == allowed(name)
                                for name, parameter in model.named_parameters()))
            self.assertEqual(model.conv1.in_channels, 4)
            self.assertTrue(groups)
            if mode != "all":
                model.train(); set_mode()
                before = [(module.running_mean.clone(), module.running_var.clone())
                          for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
                model(torch.randn(2, 4, 64, 64))
                after = [(module.running_mean, module.running_var)
                         for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
                for old, new in zip(before, after, strict=True):
                    torch.testing.assert_close(old[0], new[0])
                    torch.testing.assert_close(old[1], new[1])

        rgb_model, _, _ = resnet18.build(3, 3, "head", weights=None)
        self.assertEqual(rgb_model.conv1.in_channels, 3)

    def test_optimizer_step(self):
        model, groups, set_mode = resnet18.build(3, 2, "head", weights=None)
        before = model.fc.weight.detach().clone()
        model.train(); set_mode(); optimizer = torch.optim.AdamW(groups)
        nn.functional.cross_entropy(model(torch.randn(2, 2, 64, 64)),
                                    torch.tensor([0, 2])).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.fc.weight))


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(3, 384)
        self.blocks = nn.ModuleList([nn.Linear(384, 384), nn.Linear(384, 384)])

    def forward(self, image):
        value = self.embedding(image.mean((-2, -1)))
        for block in self.blocks:
            value = block(value)
        return value


class DinoTests(unittest.TestCase):
    def test_policies_and_optimizer_step(self):
        for mode in ("linear", "last"):
            model, groups, set_mode = dinov2.build(3, mode, backbone=FakeDino())
            trainable = {name for name, parameter in model.named_parameters()
                         if parameter.requires_grad}
            self.assertTrue(all(name.startswith(("adapter.", "head.")) or
                                (mode == "last" and name.startswith("backbone.blocks.1."))
                                for name in trainable))
            frozen = model.backbone.embedding.weight.detach().clone()
            adapter = model.adapter.weight.detach().clone()
            model.train(); set_mode(); optimizer = torch.optim.AdamW(groups)
            nn.functional.cross_entropy(model(torch.randn(2, 4, 28, 28)),
                                        torch.tensor([0, 2])).backward()
            optimizer.step()
            torch.testing.assert_close(frozen, model.backbone.embedding.weight)
            self.assertFalse(torch.equal(adapter, model.adapter.weight))
            self.assertFalse(model.backbone.training)
            if mode == "last":
                self.assertTrue(model.backbone.blocks[-1].training)


class ArtifactAndReportTests(unittest.TestCase):
    def _prepared(self):
        metadata = ({"corruption_snr_target": 0, "sample_kind": "baseline"},
                    {"corruption_snr_target": 10, "sample_kind": "gain"},
                    {"corruption_snr_target": 10, "sample_kind": "gain"})
        split = PreparedSplit(torch.randn(3, 4, 8, 8), torch.tensor([0, 1, 2]),
                              ("clean", "amp", "phase"), ("a", "b", "c"), metadata)
        controls = PreparedSplit(torch.randn(1, 4, 8, 8), torch.tensor([0]),
                                 ("noise",), ("a",),
                                 ({"corruption_snr_target": 10,
                                   "sample_kind": "increased_noise"},))
        return {"train": split, "validation": split, "noise_controls": controls}

    def test_prediction_loss_uses_supplied_class_weights(self):
        logits = torch.tensor([[2., 0., 0.], [2., 0., 0.], [2., 0., 0.]])
        labels = torch.tensor([0, 0, 1])
        split = PreparedSplit(logits, labels, ("a", "b", "c"), ("a", "b", "c"),
                              ({}, {}, {}))
        weights = torch.tensor([.5, 2., 1.])
        _, loss = predict(nn.Identity(), split, torch.device("cpu"), 2, weights)
        losses = nn.functional.cross_entropy(logits, labels, weight=weights, reduction="none")
        self.assertAlmostEqual(loss, float(losses.sum() / weights[labels].sum()))

    def test_training_artifacts_resume_and_probability_reproduction(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "job"
            model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3))
            config = {"job_id": "job", "seed": 42, "epochs": 1, "patience": 1,
                      "batch_size": 3, "normalization": {}, "weight_decay": 1e-4}
            results = train_job(model, [{"params": list(model.parameters()), "lr": .01}],
                                lambda: None, self._prepared(), (0, 1, 2), 0,
                                config, directory, "cpu")
            self.assertTrue(_complete(directory, config))
            with (directory / "predictions.csv").open() as handle:
                rows = [row for row in csv.DictReader(handle)
                        if row["split"] == "validation"]
            scores = np.asarray([[float(row[f"probability_{label}"])
                                  for label in (0, 1, 2)] for row in rows])
            reproduced = evaluate_task([int(row["true_label"]) for row in rows], scores,
                                       (0, 1, 2), self._prepared()["validation"].metadata,
                                       clean_label=0)
            self.assertAlmostEqual(reproduced["overall"]["main"]["f1"],
                                   results["validation"]["overall"]["main"]["f1"])
            self.assertEqual(results["noise_controls"]["evaluation_kind"],
                             "noise_robustness")

            expected = []
            for seed in (42, 43, 44):
                job_id = f"exp__seed{seed}"
                shutil.copytree(directory, Path(temporary) / job_id)
                expected.append({"job_id": job_id, "experiment_id": "exp", "seed": seed})
            report = write_report(Path(temporary), expected, render=False)
            report_text = report.read_text()
            self.assertIn("3/3", report_text)
            self.assertIn("Corruption-strength weighting", report_text)
            self.assertIn("Original FPR", report_text)
            self.assertIn("Increased-noise predicted counts", report_text)
            self.assertTrue((Path(temporary) /
                             "figures/model_comparison__strength.png").is_file())
            self.assertTrue((Path(temporary) /
                             "figures/model_comparison__confusion_v3.png").is_file())
            self.assertTrue((Path(temporary) /
                             "figures/model_comparison__noise.png").is_file())
            self.assertTrue((Path(temporary) / "figures/strength_weighting__v2.png").is_file())

            smoke = Path(temporary) / "smoke"
            shutil.copytree(directory, smoke / "exp__seed42")
            smoke_report = write_report(
                smoke,
                [{"job_id": "exp__seed42", "experiment_id": "exp", "seed": 42}],
                render=False,
            )
            smoke_text = smoke_report.read_text()
            headings = ["# Validation comparison", "# Confusion matrices",
                        "# Metrics by corruption strength", "# Increased-noise controls",
                        "# Hard samples", "# Training loss by epoch"]
            self.assertTrue(all(heading in smoke_text for heading in headings))
            self.assertEqual(sorted(smoke_text.index(heading) for heading in headings),
                             [smoke_text.index(heading) for heading in headings])
            self.assertIn("| `exp` | 42 |", smoke_text)
            self.assertNotIn("Three-seed comparisons", smoke_text)
            self.assertIn(".hard-samples table", smoke_text)
            self.assertEqual(smoke_text.count("![Raw and strength-weighted confusion matrices]"), 1)
            self.assertEqual(smoke_text.count("![All-model metrics by corruption strength]"), 1)
            self.assertEqual(smoke_text.count("![All-model noise-control comparison]"), 1)
            self.assertTrue((smoke /
                             "figures/model_comparison__confusion_v3.png").is_file())
            self.assertTrue((smoke / "figures/exp__loss.png").is_file())

    def test_matrix_is_deduplicated_and_seeded(self):
        matrix = experiments(("resnet18", "dinov2"), "all")
        self.assertEqual(len(matrix), len({row["experiment_id"] for row in matrix}))
        smoke_ids = {row["experiment_id"] for row in experiments(
            ("resnet18", "dinov2"), "smoke"
        )}
        self.assertEqual(smoke_ids, {
            "resnet18__head_only__d-c-r-p__base",
            "resnet18__head_and_last__d-c-r-p__base",
            "resnet18__all__d-c-r-p__base",
            "dinov2__head_adapter__d-c-r-p__base",
            "dinov2__head_last_adapter__d-c-r-p__base",
        })
        self.assertEqual({job["seed"] for job in jobs(("resnet18",), "depth")},
                         {42, 43, 44})
        self.assertEqual(len(jobs(("resnet18", "dinov2"), "smoke")), 5)
        residual = experiments(("resnet18",), "residual100")
        self.assertEqual({row["experiment_id"] for row in residual}, {
            "resnet18__head_only__r__residual100",
            "resnet18__head_and_last__r__residual100",
            "resnet18__all__r__residual100",
        })
        self.assertTrue(all(job["channels"] == ("residual",) for job in residual))
        rrr = experiments(("resnet18",), "residual_rrr100")
        self.assertEqual({row["experiment_id"] for row in rrr}, {
            "resnet18__head_only__rrr__residual100",
            "resnet18__head_and_last__rrr__residual100",
            "resnet18__all__rrr__residual100",
        })
        self.assertTrue(all(job["channels"] == RESIDUAL_RRR_CHANNELS for job in rrr))

    def test_combined_report_links_complete_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self._prepared()
            sources = []
            for index, experiment in enumerate(("old", "rrr")):
                source = root / f"source-{index}"
                job_id = f"resnet18__head_only__{experiment}__residual100__seed42"
                directory = source / job_id
                model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3))
                config = {"job_id": job_id, "experiment_id": job_id.rsplit("__seed", 1)[0],
                          "seed": 42, "epochs": 1, "patience": 1, "batch_size": 3,
                          "normalization": {}, "weight_decay": 1e-4,
                          "dataset_sha256": "same-dataset"}
                train_job(model, [{"params": list(model.parameters()), "lr": .01}],
                          lambda: None, prepared, (0, 1, 2), 0, config, directory, "cpu")
                sources.append(source)
            report = combine_runs(sources, root / "comparison", render=False)
            self.assertIn("2/2", report.read_text())
            self.assertEqual(len(list((root / "comparison").glob("resnet18*"))), 2)
            self.assertTrue(all(path.is_symlink() for path in
                                (root / "comparison").glob("resnet18*")))


if __name__ == "__main__":
    unittest.main()
