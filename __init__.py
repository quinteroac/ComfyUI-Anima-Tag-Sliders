import json
import math
import sys

import torch

import comfy.conds
import comfy.patcher_extension
import comfy.samplers
from comfy_api.latest import ComfyExtension, io

WEB_DIRECTORY = "./web"


BIAS_KEY = "anima_tag_bias"
SCALE_KEY = "anima_tag_scale"
CURVE_KEY = "anima_tag_curve"
ARRAY_KEYS = (BIAS_KEY, SCALE_KEY, CURVE_KEY)
BASE_KEY = "anima_tag_base"
CURVES_OPTION = "anima_tag_curves"
MAX_CURVE_TAGS = 4
MODE_TOOLTIP = "weight scales the tag's contribution, weight_norm does the same but keeps each cross-attention output at the magnitude of the tag as written, attention biases how much the image attends to it, guidance pushes along the difference between the prompt with and without the tag."


def extra_conds_with_tag(original, cond, ids, weights, marks, base_key):
    """marks: (array key, neutral value, apply) for each per-token array that carries the tag."""
    def extra_conds(**kwargs):
        base = original(**kwargs) if base_key is not None else None
        start = None
        if kwargs.get("prompt_type") == "positive" and kwargs.get("t5xxl_ids") is not None:
            # Same layout as writing the tag at the end of the prompt, before the adapter runs.
            prompt_ids, prompt_weights = kwargs["t5xxl_ids"], kwargs["t5xxl_weights"]
            start = len(prompt_ids) - 1
            kwargs["cross_attn"] = torch.cat((kwargs["cross_attn"], cond.to(kwargs["cross_attn"])), dim=1)
            kwargs["t5xxl_ids"] = torch.cat((prompt_ids[:-1], ids.to(prompt_ids), prompt_ids[-1:]))
            kwargs["t5xxl_weights"] = torch.cat((prompt_weights[:-1], weights.to(prompt_weights), prompt_weights[-1:]))
        out = original(**kwargs) if start is not None or base is None else base
        if base_key is not None and "c_crossattn" in out:
            keep_base(out, base, base_key)
        if "c_crossattn" in out:
            for key, neutral, apply in marks:
                mark_tag(out, key, neutral, start, len(ids), apply)
        return out
    return extra_conds


def keep_base(out, base, key):
    """Context without the tag, for the pass that leaves it out. Every prompt carries one so they still batch.
    Tags of earlier nodes sit at other positions once this tag is gone, so that pass keeps its own arrays."""
    out[key] = base["c_crossattn"]
    for array_key in ARRAY_KEYS:
        if array_key in base:
            out[key + array_key] = base[array_key]


def mark_tag(out, key, neutral, start, length, apply):
    """Writes the tag into a per-token array. Every prompt carries the array so positive and negative still batch together.
    A later node's tag comes before any guidance tag in the prompt, so it has the same positions in each pass without a guidance tag."""
    context = out["c_crossattn"].cond
    keys = [key] + [k + key for k in out if k.startswith(BASE_KEY) and not k.endswith(ARRAY_KEYS)]
    for k in keys:
        array = out[k].cond if k in out else torch.full(context.shape[:2], neutral, device=context.device, dtype=context.dtype)
        if start is not None:
            apply(array[:, start:start + length])
        out[k] = comfy.conds.CONDRegular(array)


def biased_attention(previous, bias):
    def override(func, q, k, v, *args, **kwargs):
        # Only cross-attention has the context length on its keys; self-attention is left alone.
        if k.shape[2] == bias.shape[-1]:
            kwargs["mask"] = bias[:, None, None, :].to(q)
        if previous is not None:
            return previous(func, q, k, v, *args, **kwargs)
        return func(q, k, v, *args, **kwargs)
    return override


def norm_preserving_attention(previous, scale):
    def override(func, q, k, v, *args, **kwargs):
        def attention(v):
            if previous is not None:
                return previous(func, q, k, v, *args, **kwargs)
            return func(q, k, v, *args, **kwargs)
        out = attention(v)
        if k.shape[2] != scale.shape[-1]:
            return out
        # Keys are RMS-normed, so the weight only scaled the values: v / scale is the tag as written.
        reference = attention(v / scale[:, None, :, None].to(v))
        return out * (reference.norm(dim=-1, keepdim=True) / out.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    return override


def cross_attention_wrapper(executor, *args, **kwargs):
    bias, scale = kwargs.pop(BIAS_KEY, None), kwargs.pop(SCALE_KEY, None)
    if bias is not None or scale is not None:
        options = kwargs.get("transformer_options", {}).copy()
        if bias is not None:
            options["optimized_attention_override"] = biased_attention(options.get("optimized_attention_override"), bias)
        if scale is not None:
            options["optimized_attention_override"] = norm_preserving_attention(options.get("optimized_attention_override"), scale)
        kwargs["transformer_options"] = options
    return executor(*args, **kwargs)


def pop_base_wrapper(executor, *args, **kwargs):
    for key in [k for k in kwargs if k.startswith(BASE_KEY)]:
        del kwargs[key]
    return executor(*args, **kwargs)


def without_tag(model_conds, base_key):
    out = {**model_conds, "c_crossattn": model_conds[base_key]}
    for key in ARRAY_KEYS:
        if key in model_conds:
            out[key] = model_conds[base_key + key]
    return out


def range_schedule(tags, sigma_start, sigma_end, anchor):
    """The strength inside the range; anchor: before the range the tag is off instead of as written."""
    def schedule(sigma, sample_sigmas):
        if sigma_end <= sigma <= sigma_start:
            return tags
        if anchor and sigma > sigma_start:
            return [(key, 0.0) for key, _ in tags]
        return []
    return schedule


def curve_schedule(tags, position):
    def schedule(sigma, sample_sigmas):
        x = position(sigma, sample_sigmas)
        return [(key, curve_value(points, x)) for key, points in tags]
    return schedule


def tag_guidance(previous, schedule, rescale):
    """pred = with + sum((s - 1) * (with - without_tag)): s=0 drops the tag, s=1 is the tag as written.
    schedule(sigma, sample_sigmas) gives the (tag, s) pairs of the current step."""
    def calc_cond_batch(args):
        conds, sigma = args["conds"], args["sigma"]
        active = []
        if conds[0] is not None:
            sample_sigmas = args["model_options"]["transformer_options"].get("sample_sigmas")
            active = [(key, s) for key, s in schedule(sigma.max().item(), sample_sigmas) if s != 1.0]
        extra = [[{**c, "model_conds": without_tag(c["model_conds"], key)} for c in conds[0]] for key, _ in active]
        args = {**args, "conds": conds + extra}
        if previous is not None:
            out = previous(args)
        else:
            out = comfy.samplers.calc_cond_batch(args["model"], args["conds"], args["input"], sigma, args["model_options"])
        with_tags = out[0]
        for (_, s), without in zip(active, out[len(conds):]):
            out[0] = out[0] + (s - 1.0) * (with_tags - without)
        if active and rescale > 0:
            # Same as RescaleCFG for flow models: keep the x0 std of the prompt as written.
            dims = tuple(range(1, with_tags.ndim))
            rescaled = out[0] * (with_tags.std(dim=dims, keepdim=True) / out[0].std(dim=dims, keepdim=True).clamp(min=1e-8))
            out[0] = torch.lerp(out[0], rescaled, rescale)
        return out[:len(conds)]
    return calc_cond_batch


def mode_input():
    return io.DynamicCombo.Input("mode", tooltip=MODE_TOOLTIP, options=[
        io.DynamicCombo.Option("weight", []),
        io.DynamicCombo.Option("weight_norm", []),
        io.DynamicCombo.Option("attention", []),
        io.DynamicCombo.Option("guidance", [
            io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="First step where the push applies; outside the range the tag acts as written."),
            io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Last step where the push applies."),
            io.Float.Input("rescale", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Pulls the pushed prediction back to the contrast of the prompt as written, like RescaleCFG."),
            io.Boolean.Input("anchor", default=False, tooltip="Before start_percent the image follows the prompt without the tag, so the slider starts from that image instead of the tag as written."),
        ]),
    ])


def apply_tags(model, clip, weighted, mode_values):
    mode = mode_values["mode"]
    if mode != "guidance":
        # Only guidance can push away from a tag: negative weights or attention have no learned meaning.
        weighted = [(tag, w) for tag, w in weighted if w > 0]
    patched = model.clone()
    guided = [(add_tag(patched, clip, tag, w, mode), w) for tag, w in weighted]
    if mode == "guidance":
        sampling = patched.get_model_object("model_sampling")
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, BASE_KEY, pop_base_wrapper)
        previous = patched.model_options.get("sampler_calc_cond_batch_function")
        schedule = range_schedule(guided, sampling.percent_to_sigma(mode_values["start_percent"]), sampling.percent_to_sigma(mode_values["end_percent"]), mode_values["anchor"])
        patched.model_options["sampler_calc_cond_batch_function"] = tag_guidance(previous, schedule, mode_values["rescale"])
    return patched


def encode_tag(clip, tag):
    cond, extra = clip.encode_from_tokens_scheduled(clip.tokenize(tag), show_pbar=False)[0]
    # Drop T5 </s>, the prompt already has its own.
    return cond, extra["t5xxl_ids"][:-1], extra["t5xxl_weights"][:-1]


def add_tag(patched, clip, tag, strength, mode):
    cond, ids, weights = encode_tag(clip, tag)
    marks = []
    base_key = None
    if mode == "attention":
        # log(w) on the logits multiplies the tag's attention by w before softmax renormalizes.
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, BIAS_KEY, cross_attention_wrapper)
        bias = math.log(strength)
        marks.append((BIAS_KEY, 0.0, lambda a: a.add_(bias)))
    elif mode == "guidance":
        base_key = f"{BASE_KEY}_{tag}"
    else:
        weights = weights * strength
        if mode == "weight_norm":
            patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, BIAS_KEY, cross_attention_wrapper)
            marks.append((SCALE_KEY, 1.0, lambda a: a.mul_(strength)))
    patched.add_object_patch("extra_conds", extra_conds_with_tag(patched.get_model_object("extra_conds"), cond, ids, weights, marks, base_key))
    return base_key


def curve_value(points, x):
    """Piecewise linear through the points, flat past the ends."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return points[-1][1]


def curve_step(sigma, sample_sigmas, steps):
    """Where the current step falls on a curve axis of `steps` steps; a sampler with another step count is stretched onto it."""
    sample = sample_sigmas[:-1].tolist()
    i = min(range(len(sample)), key=lambda j: abs(sample[j] - sigma))
    return i * (steps - 1) / max(len(sample) - 1, 1)


def curve_position(axis, steps, sigma_max):
    """x on a curve axis of `steps` points for the current sigma: the sampler step, or the noise level from sigma_max (x = 0) down to 0 (x = steps - 1)."""
    if axis == "noise":
        return lambda sigma, sample_sigmas: min(max(1.0 - sigma / sigma_max, 0.0), 1.0) * (steps - 1)
    return lambda sigma, sample_sigmas: curve_step(sigma, sample_sigmas, steps)


def curve_attention(previous, weight, norm, bias):
    """Per-token value scales and logit bias of the curve tags at this step. norm scales the values like weight,
    then keeps each cross-attention output at the magnitude it has with those tags as written."""
    def override(func, q, k, v, *args, **kwargs):
        def attention(v):
            if previous is not None:
                return previous(func, q, k, v, *args, **kwargs)
            return func(q, k, v, *args, **kwargs)
        if k.shape[2] != weight.shape[-1]:
            return attention(v)
        if bias is not None:
            kwargs["mask"] = bias[:, None, None, :].to(q)
        v = v * weight[:, None, :, None].to(v)
        if norm is None:
            return attention(v)
        out = attention(v * norm[:, None, :, None].to(v))
        reference = attention(v)
        return out * (reference.norm(dim=-1, keepdim=True) / out.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    return override


def curve_attention_wrapper(executor, *args, **kwargs):
    numbers = kwargs.pop(CURVE_KEY, None)
    if numbers is not None:
        options = kwargs["transformer_options"].copy()
        curves = options[CURVES_OPTION]
        sigma = options["sigmas"].max().item()
        # Row 0 is every token that is not a curve tag.
        tables = {mode: [1.0] for mode in ("weight", "weight_norm", "attention")}
        for points, position, mode in curves:
            s = max(curve_value(points, position(sigma, options["sample_sigmas"])), 0.0)
            for m, table in tables.items():
                table.append(s if m == mode else 1.0)
        index = numbers.long()
        weight, norm, attn = (torch.tensor(tables[m], device=numbers.device, dtype=numbers.dtype)[index] for m in ("weight", "weight_norm", "attention"))
        modes = {mode for _, _, mode in curves}
        options["optimized_attention_override"] = curve_attention(options.get("optimized_attention_override"), weight, norm if "weight_norm" in modes else None,
                                                                  attn.log() if "attention" in modes else None)
        kwargs["transformer_options"] = options
    return executor(*args, **kwargs)


def parse_tag(text, default_max):
    """'fisheye#4' -> ('fisheye', 4.0)."""
    tag, sep, num = text.rpartition("#")
    if sep and tag.strip():
        try:
            return tag.strip(), float(num)
        except ValueError:
            pass
    return text, default_max


def parse_line(line):
    parts = [t.strip() for t in line.split("|") if t.strip()]
    default_max = 1.5 if len(parts) <= 2 else 1.0
    return [parse_tag(t, default_max) for t in parts]


def line_weights(tags, value):
    """One tag: strength, negative (guidance only) pushes away from the tag. Two tags: -max_low..max_high, sign picks the tag.
    More: position 0..N, 0 is off, k is tag k at its max, fractions crossfade neighbours."""
    if len(tags) == 1:
        return [(tags[0][0], value)]
    if len(tags) == 2:
        return [(tags[1][0] if value > 0 else tags[0][0], abs(value))]
    i = min(math.floor(value), len(tags))
    t = value - i
    out = []
    if i >= 1:
        out.append((tags[i - 1][0], (1.0 - t) * tags[i - 1][1]))
    if i < len(tags):
        out.append((tags[i][0], t * tags[i][1]))
    return out


class AnimaTagSlider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AnimaTagSlider",
            display_name="Anima Tag Slider",
            category="Anima/conditioning",
            description="Adds a tag to the positive prompt as a model patch. Negative values use tag_low, positive values use tag_high. In guidance mode with tag_low empty, negative values push away from tag_high.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input("tag_low", default=""),
                io.String.Input("tag_high", default=""),
                mode_input(),
                io.Float.Input("strength", default=0.0, min=-1.5, max=1.5, step=0.05, display_mode=io.NumberDisplay.slider),
                io.Boolean.Input("bypass", default=False, tooltip="Returns the model unchanged, to compare against the image without the tags."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, clip, tag_low, tag_high, mode, strength, bypass):
        if bypass:
            return io.NodeOutput(model)
        if strength < 0 and not tag_low.strip() and mode["mode"] == "guidance":
            tag = tag_high.strip()
        else:
            tag, strength = (tag_high if strength > 0 else tag_low).strip(), abs(strength)
        if strength == 0 or not tag:
            return io.NodeOutput(model)
        return io.NodeOutput(apply_tags(model, clip, [(tag, strength)], mode))


class AnimaTagSliders(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AnimaTagSliders",
            display_name="Anima Tag Sliders",
            category="Anima/conditioning",
            description="One slider per line of tags. A single tag sets its strength (in guidance, negative pushes away from it), 'low | high' is a bipolar axis, three or more tags crossfade in order. 'tag#2' sets that tag's max strength.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input("tags", multiline=True, default="from below | from above\nfrom behind\ndutch angle\nperspective | fisheye"),
                io.String.Input("slider_values", default="{}"),
                mode_input(),
                io.Boolean.Input("bypass", default=False, tooltip="Returns the model unchanged, to compare against the image without the tags."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, clip, tags, slider_values, mode, bypass):
        if bypass:
            return io.NodeOutput(model)
        values = json.loads(slider_values or "{}")
        weighted = []
        for line in tags.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            weighted += line_weights(parse_line(line), float(values.get(line, 0.0)))
        weighted = [(tag, w) for tag, w in weighted if w != 0]
        if not weighted:
            return io.NodeOutput(model)
        return io.NodeOutput(apply_tags(model, clip, weighted, mode))


def parse_curves(tags, curves, steps, max_strength, mode):
    """One curve per tag line: points (step, strength) sorted by step and kept inside the axes. A tag with no curve stays as written."""
    tags = [t.strip() for t in tags.splitlines() if t.strip() and not t.strip().startswith("#")]
    if len(tags) > MAX_CURVE_TAGS:
        raise ValueError(f"Anima Tag Curves takes up to {MAX_CURVE_TAGS} tags, one per line; got {len(tags)}.")
    low = -max_strength if mode == "guidance" else 0.0
    out = []
    for tag in dict.fromkeys(tags):
        points = sorted((min(max(float(x), 0.0), steps - 1.0), min(max(float(y), low), max_strength)) for x, y in curves.get(tag) or [[0, 1.0]])
        out.append((tag, points))
    return out


class AnimaTagCurves(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AnimaTagCurves",
            display_name="Anima Tag Curves",
            category="Anima/conditioning",
            description="Up to 4 tags, one per line, each with its own strength curve over the sampling steps. Draw the curves in the editor: x is the step or the noise level (axis), y the strength (1 = the tag as written).",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input("tags", multiline=True, default="from above\nfisheye"),
                io.Int.Input("steps", default=20, min=1, max=10000, tooltip="Points on the curve axis. With axis steps they are the sampler steps, and a sampler with another step count is stretched onto the curve; with axis noise they set how finely the noise axis is divided."),
                io.Float.Input("max_strength", default=1.5, min=0.05, max=sys.float_info.max, step=0.05, tooltip="Top of the strength axis; in guidance the axis also goes down to -max_strength."),
                io.String.Input("curves", default="{}"),
                io.DynamicCombo.Input("mode", tooltip=MODE_TOOLTIP, options=[
                    io.DynamicCombo.Option("weight", []),
                    io.DynamicCombo.Option("weight_norm", []),
                    io.DynamicCombo.Option("attention", []),
                    io.DynamicCombo.Option("guidance", [
                        io.Float.Input("rescale", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Pulls the pushed prediction back to the contrast of the prompt as written, like RescaleCFG."),
                    ]),
                ]),
                io.Boolean.Input("bypass", default=False, tooltip="Returns the model unchanged, to compare against the image without the tags."),
                io.Combo.Input("axis", options=["steps", "noise"], default="steps", tooltip="What x is: the sampler step, or the noise level from full noise (left) to none (right), so a curve does the same at any schedule or step count."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, clip, tags, steps, max_strength, curves, mode, bypass, axis):
        curves = parse_curves(tags, json.loads(curves or "{}"), steps, max_strength, mode["mode"])
        if bypass or not curves:
            return io.NodeOutput(model)
        patched = model.clone()
        position = curve_position(axis, steps, float(patched.get_model_object("model_sampling").sigma_max))
        if mode["mode"] == "guidance":
            guided = []
            for tag, points in curves:
                cond, ids, weights = encode_tag(clip, tag)
                key = f"{BASE_KEY}_{tag}"
                patched.add_object_patch("extra_conds", extra_conds_with_tag(patched.get_model_object("extra_conds"), cond, ids, weights, [], key))
                guided.append((key, points))
            patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, BASE_KEY, pop_base_wrapper)
            previous = patched.model_options.get("sampler_calc_cond_batch_function")
            patched.model_options["sampler_calc_cond_batch_function"] = tag_guidance(previous, curve_schedule(guided, position), mode["rescale"])
            return io.NodeOutput(patched)
        # The tags go in as written; each step scales them in cross-attention, where weight and weight_norm act on the values only.
        options = patched.model_options.setdefault("transformer_options", {})
        registered = list(options.get(CURVES_OPTION, []))
        for tag, points in curves:
            cond, ids, weights = encode_tag(clip, tag)
            registered.append((points, position, mode["mode"]))
            number = float(len(registered))
            patched.add_object_patch("extra_conds", extra_conds_with_tag(patched.get_model_object("extra_conds"), cond, ids, weights, [(CURVE_KEY, 0.0, lambda a, n=number: a.fill_(n))], None))
        options[CURVES_OPTION] = registered
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, CURVE_KEY, curve_attention_wrapper)
        return io.NodeOutput(patched)


class AnimaTagSlidersExtension(ComfyExtension):
    async def get_node_list(self):
        return [AnimaTagSlider, AnimaTagSliders, AnimaTagCurves]


async def comfy_entrypoint():
    return AnimaTagSlidersExtension()
