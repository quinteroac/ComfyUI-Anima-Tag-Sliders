import json
import math

import torch

import comfy.conds
import comfy.patcher_extension
import comfy.samplers
from comfy_api.latest import ComfyExtension, io

WEB_DIRECTORY = "./web"


BIAS_KEY = "anima_tag_bias"
SCALE_KEY = "anima_tag_scale"
BASE_KEY = "anima_tag_base"
MODE_TOOLTIP = "weight scales the tag's contribution, weight_norm does the same but keeps each cross-attention output at the magnitude of the tag as written, attention biases how much the image attends to it, guidance pushes along the difference between the prompt with and without the tag."


def extra_conds_with_tag(original, cond, ids, weights, attn_bias, value_scale, base_key):
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
        if attn_bias is not None and "c_crossattn" in out:
            mark_tag(out, BIAS_KEY, 0.0, start, len(ids), lambda a: a.add_(attn_bias))
        if value_scale is not None and "c_crossattn" in out:
            mark_tag(out, SCALE_KEY, 1.0, start, len(ids), lambda a: a.mul_(value_scale))
        return out
    return extra_conds


def keep_base(out, base, key):
    """Context without the tag, for the pass that leaves it out. Every prompt carries one so they still batch.
    Tags of earlier nodes sit at other positions once this tag is gone, so that pass keeps its own arrays."""
    out[key] = base["c_crossattn"]
    for array_key in (BIAS_KEY, SCALE_KEY):
        if array_key in base:
            out[key + array_key] = base[array_key]


def mark_tag(out, key, neutral, start, length, apply):
    """Writes the tag into a per-token array. Every prompt carries the array so positive and negative still batch together.
    A later node's tag comes before any guidance tag in the prompt, so it has the same positions in each pass without a guidance tag."""
    context = out["c_crossattn"].cond
    keys = [key] + [k + key for k in out if k.startswith(BASE_KEY) and not k.endswith((BIAS_KEY, SCALE_KEY))]
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
    for key in (BIAS_KEY, SCALE_KEY):
        if key in model_conds:
            out[key] = model_conds[base_key + key]
    return out


def tag_guidance(previous, tags, sigma_start, sigma_end, rescale, anchor):
    """pred = with + sum((s - 1) * (with - without_tag)): s=0 drops the tag, s=1 is the tag as written.
    anchor: before the range the tag is off instead of as written."""
    def calc_cond_batch(args):
        conds, sigma = args["conds"], args["sigma"]
        current = sigma.max().item()
        active = []
        if conds[0] is not None and sigma_end <= current <= sigma_start:
            active = [(key, s) for key, s in tags if s != 1.0]
        elif conds[0] is not None and anchor and current > sigma_start:
            active = [(key, 0.0) for key, _ in tags]
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
        patched.model_options["sampler_calc_cond_batch_function"] = tag_guidance(previous, guided, sampling.percent_to_sigma(mode_values["start_percent"]), sampling.percent_to_sigma(mode_values["end_percent"]),
                                                                                 mode_values["rescale"], mode_values["anchor"])
    return patched


def add_tag(patched, clip, tag, strength, mode):
    cond, extra = clip.encode_from_tokens_scheduled(clip.tokenize(tag), show_pbar=False)[0]
    # Drop T5 </s>, the prompt already has its own.
    ids, weights = extra["t5xxl_ids"][:-1], extra["t5xxl_weights"][:-1]
    attn_bias = value_scale = base_key = None
    if mode == "attention":
        # log(w) on the logits multiplies the tag's attention by w before softmax renormalizes.
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, BIAS_KEY, cross_attention_wrapper)
        attn_bias = math.log(strength)
    elif mode == "guidance":
        base_key = f"{BASE_KEY}_{tag}"
    else:
        weights = weights * strength
        if mode == "weight_norm":
            patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, BIAS_KEY, cross_attention_wrapper)
            value_scale = strength
    patched.add_object_patch("extra_conds", extra_conds_with_tag(patched.get_model_object("extra_conds"), cond, ids, weights, attn_bias, value_scale, base_key))
    return base_key


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


class AnimaTagSlidersExtension(ComfyExtension):
    async def get_node_list(self):
        return [AnimaTagSlider, AnimaTagSliders]


async def comfy_entrypoint():
    return AnimaTagSlidersExtension()
