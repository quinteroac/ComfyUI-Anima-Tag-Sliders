# ComfyUI Anima Tag Sliders

Tag sliders for [Anima](https://huggingface.co/circlestone-labs/Anima) in ComfyUI. Each slider adds a Danbooru tag to the positive prompt as a `MODEL` patch, so you can scale it up, fade it in or push away from it, and chain nodes like LoRAs.

The tag goes into the prompt before Anima's LLM adapter, the same way it would if you wrote it at the end of the prompt. A strength of 1 looks like the tag as written.

## Nodes

**Anima Tag Slider**: one bipolar slider. Negative strength uses `tag_low` and positive strength uses `tag_high`. In guidance mode with `tag_low` empty, negative strength pushes away from `tag_high`.

**Anima Tag Sliders**: one slider per line of `tags`.

- `big eyes`: a single tag, strength 0 to 1.5. In guidance mode it goes down to −1.5, which pushes away from the tag.
- `from below | from above`: a bipolar axis, −1.5 to 1.5.
- `chibi | petite | tall`: three or more tags crossfade in order. 0 is off, and k is tag k at full strength.
- `fisheye#4`: sets that tag's maximum strength.
- Lines starting with `#` are skipped.

Both nodes have a `bypass` switch that returns the model unchanged, so you can compare against the image without the tags.

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
