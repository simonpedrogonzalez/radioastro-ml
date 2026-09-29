import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from ml.cnn import Split, augment, configure_stage, parity_features, preload, preprocess, residual_channels, residual_scale, support_mask, train


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.image = torch.randn(4, 256, 256)

    def test_parity_actual_centre_and_full_field(self):
        y, x = torch.meshgrid(torch.arange(256) - 128, torch.arange(256) - 128, indexing="ij")
        even, odd = (x*x + y*y).float(), (x + y).float()
        channels = residual_channels(even + odd, "parity")
        torch.testing.assert_close(channels[1, 1:, 1:], even[1:, 1:])
        torch.testing.assert_close(channels[2, 1:, 1:], odd[1:, 1:])
        self.assertEqual(channels[1:, 0, :].count_nonzero(), 0)
        self.assertEqual(channels[1:, :, 0].count_nonzero(), 0)
        torch.testing.assert_close(channels[0], even + odd)
        transformed = augment(channels[None])[0]
        # Every channel gets exactly the same square symmetry.
        self.assertTrue(any(torch.equal(transformed, torch.rot90(channels, k, (-2, -1)).flip(-1) if flip else torch.rot90(channels, k, (-2, -1)))
                            for k in range(4) for flip in (False, True)))

    def test_scale_sign_and_no_crop(self):
        output, fraction = preprocess(self.image, "parity")
        scaled, _ = preprocess(self.image * 17, "parity")
        torch.testing.assert_close(output, scaled, atol=2e-6, rtol=2e-6)
        self.assertEqual(output.shape, (3, 224, 224))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(fraction, 0)
        changed = self.image.clone()
        changed[2, :20] += 1  # Well outside the scale annulus and radius-80 disk.
        self.assertFalse(torch.equal(output, preprocess(changed, "parity")[0]))
        with self.assertRaisesRegex(ValueError, "MAD scale"):
            preprocess(torch.ones_like(self.image), "parity")
        # Shared scale, signed even/odd values; check against the defining equations.
        y, x = np.indices((256, 256))
        r = np.hypot(x-128, y-128)
        b = self.image[2].numpy()[(r >= 32) & (r < 72)]
        s = 1.4826 * np.median(abs(b - np.median(b)))
        raw = residual_channels(self.image[2], "parity")
        z = (torch.asinh((raw/s).clamp(-10, 10))/np.arcsinh(10) + 1)/2
        z = torch.nn.functional.interpolate(z[None], (224, 224), mode="bilinear", align_corners=False, antialias=True)[0]
        expected = (z - torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]
        torch.testing.assert_close(output, expected)

    def test_parity_features_are_raw_paired_energies(self):
        y, x = np.indices((256, 256)) - 128
        even, odd = (x*x + y*y).astype(float), (x-y).astype(float)
        self.image[2] = torch.from_numpy(even + odd).float()
        scale = residual_scale(self.image[2])
        expected = [np.log1p(np.mean((a[1:, 1:]/scale)**2)) for a in (even, odd)]
        np.testing.assert_allclose(parity_features(self.image), expected, rtol=1e-10)
        np.testing.assert_allclose(parity_features(self.image * 17), expected, rtol=1e-6)

    def test_support_and_source_filter(self):
        self.assertFalse(support_mask(self.image).any())
        circular = self.image.clone()
        circular[:3, :10] = 0
        self.assertTrue(support_mask(circular).any())
        hole = self.image.clone()
        hole[:3, 50, 50] = 0
        with self.assertRaisesRegex(ValueError, "interior"):
            support_mask(hole)
        for value in (float("nan"), float("inf")):
            hole[0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, "finite"):
                support_mask(hole)
        items = [dict(image=image, sample_id=f"{source}_{label}", label=label,
                      label_metadata={}, qa={}, paths={"products": {"clean": "unused"}})
                 for source, image in (("0005+383", self.image), ("0006-063", circular))
                 for label in range(3)]

        class FakeDataset(list):
            samples = [SimpleNamespace(sample_id=s["sample_id"]) for s in items]

        with patch("ml.cnn.FitsSimulationDataset", return_value=FakeDataset(items)), patch("ml.cnn.fits.getheader", return_value={"CRPIX1": 129, "CRPIX2": 129}):
            split = preload(Path("unused.json"), "train", "parity")
            self.assertEqual(split.audit["retained_samples"], 3)
            self.assertEqual(split.audit["excluded_samples"], 3)
            self.assertEqual(split.audit["excluded_sources"], 1)
            self.assertTrue(all(s.startswith("0005+383_") for s in split.batch["sample_id"]))
            items[4]["image"] = self.image
            with self.assertRaisesRegex(ValueError, "Mixed support"):
                preload(Path("unused.json"), "train", "parity")
        with patch("ml.cnn.FitsSimulationDataset", return_value=FakeDataset(items)), patch("ml.cnn.fits.getheader", return_value={"CRPIX1": 128, "CRPIX2": 128}):
            with self.assertRaisesRegex(ValueError, "reference pixel"):
                preload(Path("unused.json"), "train", "parity")


class FreezingTests(unittest.TestCase):
    def test_head_only_keeps_backbone_frozen_through_second_stage(self):
        torch.manual_seed(42)
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 3)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        batch = {"label": torch.tensor([0, 1, 2]), "sample_id": ["clean", "amp", "phase"],
                 "label_metadata": [
                     {"corruption_snr_target": 0, "sample_kind": "baseline"},
                     {"corruption_snr_target": 10, "sample_kind": "gain"},
                     {"corruption_snr_target": 10, "sample_kind": "gain"},
                 ]}
        split = Split(torch.randn(3, 3, 64, 64), batch, {})
        history, settings = train(model, split, split, "cpu", warmup_epochs=1, finetune_epochs=1, head_only=True)
        self.assertEqual([h["stage"] for h in history], ["head", "head_continued"])
        self.assertEqual([h["trainable_parameters"] for h in history], [1539, 1539])
        self.assertTrue(settings["head_only"])
        for key, value in model.state_dict().items():
            if not key.startswith("fc."):
                self.assertTrue(torch.equal(before[key], value), key)
        self.assertFalse(torch.equal(before["fc.weight"], model.fc.weight))

    def test_only_allowed_weights_change_and_bn_stays_frozen(self):
        model = resnet18(weights=None)  # Unit test never downloads weights.
        model.fc = nn.Linear(512, 3)
        for finetune, expected in ((False, 1539), (True, 4720131)):
            configure_stage(model, finetune)
            allowed = {n for n, p in model.named_parameters() if p.requires_grad}
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), expected)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 2])).backward()
            optimizer.step()
            after = model.state_dict()
            for key in before.keys() - allowed:
                self.assertTrue(torch.equal(before[key], after[key]), key)
            self.assertFalse(torch.equal(before["fc.weight"], after["fc.weight"]))
            if finetune:
                self.assertFalse(torch.equal(before["layer4.1.conv1.weight"], after["layer4.1.conv1.weight"]))


if __name__ == "__main__":
    unittest.main()
