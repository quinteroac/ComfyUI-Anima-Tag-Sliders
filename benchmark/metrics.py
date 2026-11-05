"""Per-image metrics for the benchmark, written to metrics.jsonl next to the manifest. Resumable.

python metrics.py --root /path/to/output/anima_tag_bench --phase phase1 [--device cpu]

Per image:
  wd_tag        WD v1.4 probability of the tested tag (None when the tagger has no such tag)
  wd_base       mean WD probability of the base prompt's own tags (how much of the prompt survives)
  wd_mono       max WD probability of monochrome / greyscale
  siglip_text   SigLIP2 cosine between the image and "an anime illustration, <tag>"
  colorfulness  Hasler-Susstrunk colorfulness of the RGB image
  hf_energy     mean |Laplacian| of the grayscale image in [0, 1] (noise / detail)
  embedding     SigLIP2 image embedding, kept for drift and saturation in the report
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

COMFY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WD_DIR = os.path.join(COMFY, "models", "booru_tagger", "wd-v1-4-moat-tagger-v2")
SIGLIP_DIR = os.path.join(COMFY, "models", "siglip2", "siglip2-base-patch16-512")
BASE_TAGS = {
    "P1": ["1girl", "school_uniform", "city", "standing", "full_body"],
    "P2": ["1girl", "school_uniform", "upper_body", "looking_at_viewer"],
    "P3": ["no_humans", "scenery", "park"],
    "PA": ["1girl", "school_uniform", "city", "full_body"],
    "PQ": ["1girl", "school_uniform", "city", "standing", "full_body"],
    "P4": ["1girl", "school_uniform", "city", "standing"],
    "AE": ["1girl", "school_uniform", "city", "standing", "full_body"],
}


class WDTagger:
    def __init__(self, device):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(os.path.join(WD_DIR, "model.onnx"), providers=providers)
        self.size = self.session.get_inputs()[0].shape[1]
        with open(os.path.join(WD_DIR, "selected_tags.csv")) as f:
            self.index = {row["name"]: i for i, row in enumerate(csv.DictReader(f))}

    def __call__(self, image):
        w, h = image.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(image, ((side - w) // 2, (side - h) // 2))
        x = np.asarray(canvas.resize((self.size, self.size), Image.BICUBIC), dtype=np.float32)[:, :, ::-1]
        probs = self.session.run(None, {self.session.get_inputs()[0].name: x[None].copy()})[0][0]
        return lambda tag: float(probs[self.index[tag]]) if tag in self.index else None


def colorfulness(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def hf_energy(rgb):
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return float(np.abs(cv2.Laplacian(gray, cv2.CV_32F)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    folder = os.path.join(args.root, args.phase)
    out_path = os.path.join(folder, "metrics.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {e["id"] for e in map(json.loads, f)}
    with open(os.path.join(folder, "manifest.jsonl")) as f:
        entries = [e for e in map(json.loads, f) if e["status"] == "success" and e["id"] not in done]
    print(f"{len(done)} done, {len(entries)} to score", flush=True)

    wd = WDTagger(args.device)
    siglip = AutoModel.from_pretrained(SIGLIP_DIR).to(args.device).eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(SIGLIP_DIR)

    for e in entries:
        image = Image.open(os.path.join(args.root, e["file"])).convert("RGB")
        rgb = np.asarray(image, dtype=np.float32)
        probs = wd(image)
        tag_key = e["tag"].lstrip("@").replace(" ", "_")
        base = [p for p in (probs(t) for t in BASE_TAGS[e["base"]]) if p is not None]
        inputs = processor(text=[f"an anime illustration, {e['tag']}"], images=[image], padding="max_length", max_length=64, return_tensors="pt").to(args.device)
        img = siglip.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output
        txt = siglip.get_text_features(input_ids=inputs["input_ids"]).pooler_output
        img = torch.nn.functional.normalize(img.float(), dim=-1)
        txt = torch.nn.functional.normalize(txt.float(), dim=-1)
        row = {
            "id": e["id"],
            "wd_tag": probs(tag_key),
            "wd_base": float(np.mean(base)) if base else None,
            "wd_mono": max(probs("monochrome") or 0, probs("greyscale") or 0),
            "siglip_text": float((img * txt).sum()),
            "colorfulness": colorfulness(rgb),
            "hf_energy": hf_energy(rgb),
            "embedding": [round(v, 5) for v in img[0].tolist()],
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
