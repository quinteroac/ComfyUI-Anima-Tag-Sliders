import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import comfy.model_base  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.supported_models  # noqa: E402

spec = importlib.util.spec_from_file_location("anima_tag_sliders", Path(__file__).resolve().parents[1] / "__init__.py")
sliders = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sliders)

DEVICE = torch.device("cpu")
SIGMAS = torch.tensor([1.0, 0.8, 0.5, 0.0])


class FakeClip:
    """One Qwen state and one T5 id per word, plus </s>."""
    def __init__(self):
        self.generator = torch.Generator().manual_seed(7)
        self.encoded = {}

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, text, show_pbar=True):
        if text not in self.encoded:
            n = len(text.split()) + 1
            ids = torch.randint(2, 1000, (n,), generator=self.generator)
            ids[-1] = 1
            self.encoded[text] = (torch.randn((1, n, 1024), generator=self.generator), ids)
        cond, ids = self.encoded[text]
        return [[cond, {"t5xxl_ids": ids, "t5xxl_weights": torch.ones(len(ids))}]]


def recording_sampler(outputs):
    """Calls the model once per step on the starting latent, so each step's prediction can be compared on its own."""
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        for sigma in sigmas[:-1]:
            outputs.append(model(x, sigma * x.new_ones([x.shape[0]]), **extra_args))
        return x
    return comfy.samplers.KSAMPLER(sample)


class CurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = comfy.supported_models.Anima(dict(
            image_model="anima", max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
            concat_padding_mask=True, model_channels=64, num_blocks=2, num_heads=2, crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
            pos_emb_interpolation="crop", use_adaln_lora=True, adaln_lora_dim=16, extra_per_block_abs_pos_emb=False, rope_h_extrapolation_ratio=4.0,
            rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0, rope_enable_fps_modulation=False, dtype=torch.float32))
        config.custom_operations = comfy.ops.manual_cast
        model = comfy.model_base.Anima(config, device=DEVICE)
        generator = torch.Generator().manual_seed(42)
        state = {}
        for name, value in model.state_dict().items():
            if not name.startswith("diffusion_model.") or not value.is_floating_point():
                continue
            state[name] = torch.ones_like(value) if "norm" in name else torch.randn(value.shape, generator=generator) * 0.1
        model.load_state_dict(state, strict=False)
        cls.model = comfy.model_patcher.ModelPatcher(model, load_device=DEVICE, offload_device=torch.device("cpu"))
        cls.clip = FakeClip()
        cls.positive = cls.clip.encode_from_tokens_scheduled("one girl standing in a field")
        cls.negative = cls.clip.encode_from_tokens_scheduled("worst quality")
        cls.noise = torch.randn((1, 16, 1, 8, 8), generator=generator)

    @classmethod
    def tearDownClass(cls):
        comfy.model_management.unload_all_models()

    def predictions(self, model):
        outputs = []
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(self.positive, self.negative)
        guider.set_cfg(2.0)
        with torch.inference_mode():
            guider.sample(self.noise.clone(), torch.zeros_like(self.noise), recording_sampler(outputs), SIGMAS, disable_pbar=True, seed=1)
        return outputs

    def slider(self, tag, strength, mode):
        return sliders.AnimaTagSlider.execute(self.model, self.clip, "", tag, {"mode": mode, "start_percent": 0.0, "end_percent": 1.0, "rescale": 0.0, "anchor": False}, strength, False)[0]

    def curves(self, curves, mode, steps=3):
        return sliders.AnimaTagCurves.execute(self.model, self.clip, "\n".join(curves), steps, 4.0, json.dumps(curves), {"mode": mode, "rescale": 0.0}, False)[0]

    def test_flat_curve_is_the_slider(self):
        for mode in ("weight", "weight_norm", "attention", "guidance"):
            with self.subTest(mode=mode):
                for curve, slider in zip(self.predictions(self.curves({"fisheye lens": [[0, 2.0]]}, mode)), self.predictions(self.slider("fisheye lens", 2.0, mode))):
                    torch.testing.assert_close(curve, slider, rtol=0, atol=1e-4)

    def test_each_step_reads_its_curve(self):
        curve = {"fisheye lens": [[0, 3.0], [2, 1.5]]}
        for mode in ("weight_norm", "attention", "guidance"):
            with self.subTest(mode=mode):
                out = self.predictions(self.curves(curve, mode))
                torch.testing.assert_close(out[0], self.predictions(self.slider("fisheye lens", 3.0, mode))[0], rtol=0, atol=1e-4)
                torch.testing.assert_close(out[1], self.predictions(self.slider("fisheye lens", 2.25, mode))[1], rtol=0, atol=1e-4)
                torch.testing.assert_close(out[2], self.predictions(self.slider("fisheye lens", 1.5, mode))[2], rtol=0, atol=1e-4)

    def test_curve_at_one_is_the_tag_as_written(self):
        written = self.predictions(self.slider("fisheye lens", 1.0, "weight"))
        for mode in ("weight", "weight_norm", "attention", "guidance"):
            with self.subTest(mode=mode):
                for curve, slider in zip(self.predictions(self.curves({"fisheye lens": [[0, 1.0]]}, mode)), written):
                    torch.testing.assert_close(curve, slider, rtol=0, atol=1e-4)

    def test_tags_have_their_own_curves(self):
        mixed = self.predictions(self.curves({"fisheye lens": [[0, 2.0]], "from above": [[0, 1.0]]}, "attention"))
        swapped = self.predictions(self.curves({"fisheye lens": [[0, 1.0]], "from above": [[0, 2.0]]}, "attention"))
        self.assertGreater((mixed[0] - swapped[0]).abs().max().item(), 1e-3)
        # Two chained curve nodes number their tags apart.
        chained = sliders.AnimaTagCurves.execute(self.curves({"fisheye lens": [[0, 2.0]]}, "attention"), self.clip, "from above", 3, 4.0, json.dumps({"from above": [[0, 1.0]]}),
                                                 {"mode": "attention"}, False)[0]
        for a, b in zip(self.predictions(chained), mixed):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-4)

    def test_curve_helpers(self):
        points = [(1.0, 0.0), (3.0, 2.0)]
        self.assertEqual([sliders.curve_value(points, x) for x in (0, 1, 2, 3, 5)], [0.0, 0.0, 1.0, 2.0, 2.0])
        sample = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        self.assertEqual([sliders.curve_step(s, sample, 7) for s in (1.0, 0.75, 0.25)], [0.0, 2.0, 6.0])
        parsed = sliders.parse_curves("a\n# skipped\nb\na", {"a": [[9, 5.0], [-1, -2.0]]}, 4, 1.5, "weight")
        self.assertEqual(parsed, [("a", [(0.0, 0.0), (3.0, 1.5)]), ("b", [(0, 1.0)])])
        self.assertEqual(sliders.parse_curves("a", {"a": [[0, -2.0]]}, 4, 1.5, "guidance"), [("a", [(0.0, -1.5)])])
        with self.assertRaises(ValueError):
            sliders.parse_curves("a\nb\nc\nd\ne", {}, 4, 1.5, "weight")


if __name__ == "__main__":
    unittest.main()
