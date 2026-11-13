# ComfyUI Anima Tag Sliders

Tag sliders for [Anima](https://huggingface.co/circlestone-labs/Anima) in ComfyUI. Each slider adds a Danbooru tag to the positive prompt as a `MODEL` patch, so you can scale it up, fade it in or push away from it, and chain nodes like LoRAs.

The tag goes into the prompt before Anima's LLM adapter, the same way it would if you wrote it at the end of the prompt. A strength of 1 looks like the tag as written.

## Install

You need a recent ComfyUI. Clone the repo into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/quinteroac/ComfyUI-Anima-Tag-Sliders
```

You can also use ComfyUI-Manager: **Install via Git URL** with the repo URL.

The nodes need no extra Python packages. Only `benchmark/metrics.py` does (see Benchmark).

To update later, run `git pull` in the pack folder and restart ComfyUI.

## Setup

Use the usual Anima files:

- the Anima model in `models/diffusion_models`;
- the Qwen3 0.6B text encoder in `models/text_encoders`;
- the Qwen-Image VAE in `models/vae`.

Wire the graph like this:

1. **Load Diffusion Model** → **Anima Tag Sliders** (`model`) → KSampler.
2. **Load CLIP** (type `stable_diffusion`) → **Anima Tag Sliders** (`clip`). Use the same CLIP that encodes your prompts.
3. Leave the slider tags out of the prompt, because the node adds them. Set a strength on the node's sliders and queue.

You can find the nodes under `Anima/conditioning`. Chain several nodes like LoRAs, each with its own mode.

## Nodes

**Anima Tag Slider**: one bipolar slider. Negative strength uses `tag_low` and positive strength uses `tag_high`. In guidance mode with `tag_low` empty, negative strength pushes away from `tag_high`.

**Anima Tag Sliders**: one slider per line of `tags`.

- `big eyes`: a single tag, strength 0 to 1.5. In guidance mode it goes down to −1.5, which pushes away from the tag.
- `from below | from above`: a bipolar axis, −1.5 to 1.5.
- `chibi | petite | tall`: three or more tags crossfade in order. 0 is off, and k is tag k at full strength.
- `fisheye#4`: sets that tag's maximum strength.
- Lines starting with `#` are skipped.

**Anima Tag Curves**: up to 4 tags, one per line, each with its own strength curve over the sampling steps. The editor in the node draws one curve per tag: x is the step (0 to `steps` − 1), y the strength (0 to `max_strength`, and down to −`max_strength` in guidance). The dashed line at 1 is the tag as written.

- Click a tag in the legend to edit its curve. The others stay visible, faded.
- Click to add a point, drag to move it, double-click or right-click to delete it.
- Between points the strength is linear, past the first and last point it stays flat. A new tag starts flat at 1.
- Changing `steps` keeps the points at the same place on the curve. A sampler with a different step count is stretched onto the curve.
- Modes are the same as the sliders. Guidance only has `rescale`, since the curve replaces `start_percent`, `end_percent` and `anchor`; a curve at 0 early on is the anchor.
- The tags go in as written; each step scales them in cross-attention. weight and weight_norm scale only the values there, which in Anima matches scaling the token weights. At strength 0 the tag's values are zero, like padding, not removed.

`example_workflows/anima_tag_curves.json` is a text-to-image workflow on anima-base with `fisheye` at 3 for the first 10 steps, fading to 1 by step 20.

All nodes have a `bypass` switch that returns the model unchanged, so you can compare against the image without the tags.

## Modes

| Mode | What it does |
|---|---|
| weight | Scales the tag's token weights. |
| weight_norm | Weight, plus each cross-attention output is kept at the magnitude it has with the tag as written. |
| attention | Adds log(w) to the attention logits on the tag's tokens. |
| guidance | One extra model pass per tag: `pred = with + (s − 1) · (with − without_tag)`. |

`guidance` has these extra options:

- `start_percent` and `end_percent`: the step range where the push applies.
- `rescale`: pulls the prediction back to the contrast of the prompt as written, like RescaleCFG.
- `anchor`: before `start_percent` the tag is off instead of as written, so the image starts from the prompt without it.

Only guidance accepts negative strengths. In the other modes, negative values have no learned meaning and give noise.

## Which mode

These come from the benchmark in `benchmark/` (43 tags, anima-base, 4 seeds on the main settings).

- Camera and lens tags (`fisheye`, `from below`): `weight_norm`. It stays clean and in color up to strength 16.
- `weight` and `attention`: they behave alike up to about 4. From 8 the tag is read literally, so `fisheye` becomes a giant eye.
- Local content tags (`big eyes`, `hat`, `cat`): `guidance` from `start_percent` 0.35, at around strength 4.
- Keeping the rest of the image: turn on `anchor`. This works for small additions such as a cat, a phone or a hat, but not for a whole outfit.
- Pushing away from a tag: guidance with a negative strength. For example, `big eyes` −4 from 0.35 narrows the eyes. A trait the prompt itself asks for is harder to remove: `smile` −8 removed it on only 2 of 4 seeds.
- Guidance strengthens everything the tag correlates with in the training data, not just the tag. If a tag turns the image gray, add `monochrome, greyscale` to the negative prompt.

## Tests

From the ComfyUI root:

```bash
python custom_nodes/ComfyUI-Anima-Tag-Sliders/tests/test_curves.py
```

It samples a tiny random Anima on CPU and checks that a flat curve equals the slider node at the same strength in every mode, and that each step uses its own value of the curve.

## Benchmark

`benchmark/` holds the scripts that produced those numbers:

- `spec.py` lists the jobs.
- `run.py` queues them on a running ComfyUI. Use a separate instance and pass its `--url`.
- `metrics.py` scores each image with the WD v1.4 MOAT tagger (`models/booru_tagger/wd-v1-4-moat-tagger-v2`) and SigLIP2 (`models/siglip2/siglip2-base-patch16-512`). It needs `onnxruntime`, `transformers` and `opencv-python`.
- `analyze.py` aggregates the scores, and `sheets.py` builds the contact sheets.

`related_work.md` lists the published precedent for each mode.

```bash
python benchmark/run.py --url http://127.0.0.1:8189 --root /path/to/output/anima_tag_bench --phase phase1
python benchmark/metrics.py --root /path/to/output/anima_tag_bench --phase phase1
python benchmark/analyze.py --root /path/to/output/anima_tag_bench
```

## License

MIT.
