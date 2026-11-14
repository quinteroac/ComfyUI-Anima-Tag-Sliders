from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_curves  # noqa: E402

import comfy.model_management  # noqa: E402
import comfy.samplers  # noqa: E402

sliders = test_curves.sliders


class StyleCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = test_curves.CurveTests
        base.setUpClass()
        cls.model, cls.clip, cls.positive, cls.negative, cls.noise = base.model, base.clip, base.positive, base.negative, base.noise

    @classmethod
    def tearDownClass(cls):
        comfy.model_management.unload_all_models()

    def sample(self, model):
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(self.positive, self.negative)
        guider.set_cfg(3.0)
        result = guider.sample(self.noise.clone(), torch.zeros_like(self.noise), comfy.samplers.ksampler("euler"), torch.tensor([1.0, 0.6, 0.3, 0.0]), disable_pbar=True, seed=42)
        self.assertTrue(torch.isfinite(result).all())
        return result

    def code(self, code, mix="", strength=1.5):
        return sliders.AnimaStyleCode.execute(self.model, self.clip, code, mix, 2, 1, 5000, strength, False)

    def test_codes_pick_repeatable_styles(self):
        self.assertEqual(sliders.code_styles(7, 2, 1, 5000), sliders.code_styles(7, 2, 1, 5000))
        self.assertNotEqual(sliders.code_styles(7, 2, 1, 5000), sliders.code_styles(8, 2, 1, 5000))
        picked = sliders.code_styles(7, 2, 2, 10)
        self.assertAlmostEqual(sum(w for _, w in picked), 4.0, places=6)
        artists = [s for s, _ in picked if s["category"] == "artist"]
        self.assertEqual(len(artists), 2)
        self.assertTrue(all(a in sliders.ARTISTS[:10] and a["text"].startswith("@") for a in artists))
        self.assertEqual(sliders.code_styles(7, 0, 0, 10), [])

    def test_code_changes_the_image(self):
        model, listing = self.code(7)
        self.assertEqual(len(listing.splitlines()), 3)
        self.assertGreater((self.sample(model) - self.sample(self.model)).abs().max().item(), 1e-3)
        self.assertIs(self.code(7, strength=0)[0], self.model)


if __name__ == "__main__":
    unittest.main()
