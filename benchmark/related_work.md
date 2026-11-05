# Related work and scope

Notes for the benchmark report. Every mode of the node has a published precedent. What is specific to this work is applying it to Anima and measuring it by tag group.

## References by mode

| Mode | Precedent | Difference from our implementation |
|---|---|---|
| weight | A1111 / Compel `(tag:w)` prompt weighting. The Anima model card says it needs more weight than SDXL, e.g. `(chibi:2)` [1] | The weight is applied to `t5xxl_weights`, before the LLM adapter |
| weight_norm | A1111 renormalization: after weighting, the embedding is rescaled to its original mean ("mean" options of Advanced CLIP Text Encode) [2] | We normalize each cross-attention output to the norm it has with the tag at weight 1, not the embedding |
| attention | Prompt-to-Prompt reweighting [3]. It is an editing method: an "equalizer" multiplies one token's attention map and keeps the rest of the source image | We generate from scratch, with no source image. We also add log(w) to the logits, so the softmax renormalizes |
| guidance | SEGA: per-concept guidance, with an invertible direction and warmup [4]. Composable Diffusion | Equivalent to per-tag SEGA with `start_percent`, without its momentum or per-dimension threshold |
| guidance + rescale | RescaleCFG [5]; APG splits the parallel and orthogonal components and reports less oversaturation than Rescale [6] | We use the RescaleCFG rescale for flow, applied to the x0 of the prompt as written |
| guidance + anchor | A1111 prompt editing `[:tag:when]`, which adds a tag partway through sampling | Before `start_percent` the tag is off; after it the guidance push applies, so strength still scales the tag |
| guidance + lock | MasaCtrl, mutual self-attention control [15]; Prompt-to-Prompt [3] | A reference without the tags follows the tagged latent with the sampler's noise, and the tagged passes attend to its keys and values next to their own instead of in their place, which would drop the tag |

Sliders in general:
- **Concept Sliders** [7]: a LoRA trained per concept, designed to interfere little with other attributes.
- **FreeSliders** [8]: estimates Concept Sliders at inference time, without training. Defines three properties of a good slider: *range*, *smoothness* and *preservation*. They match our adherence, saturation and base-prompt preservation measures.
- **Prompt Sliders** [9] and **Baumann et al.** [10]: directions in the text embedding, learned or taken from contrastive prompt pairs. They work on CLIP, with no adapter in between.

Disentangling concepts (tags come "bundled", e.g. fisheye → gray):
- **Perp-Neg** [11]: uses only the component of the negative perpendicular to the main prompt.
- **APG** [6]: attributes oversaturation to the parallel component of the guidance.
- **AdaVD** [12]: orthogonal complement in the cross-attention value space, applied to concept erasure.
- **FluxSpace** [13]: linear editing on the attention outputs of flow transformers, where semantics are more disentangled.
- **SliderSpace** [14]: finds mutually orthogonal slider directions.

## Applicable to other models

- **Benchmark design.** "No tag" and "written tag" controls, a strength sweep per mode, concept groups, one seed to explore and several to confirm.
- **Questions and metrics.** How hard each mode pushes, where it breaks, how it breaks (literal reading, style collapse, noise) and what the concept drags along.
- **Guidance mode.** It only needs the predictions with and without the tag, so it does not depend on the architecture. It may behave differently on distilled models or at CFG 1 (`x_turbo` gives a first sample).
- **Rescale and the disentangling projection.** They operate on the predictions, like guidance.
- **Step window.** Composition being fixed in the first steps (start ≈ 0.35 breaks less) is, in principle, a general property of diffusion and flow. It should be verified per model.
- **Concept correlations.** They come from the dataset, not the architecture. The method to detect them, comparing the tagger or a VLM with and without the tag, transfers. The specific correlations do not.

What has to be adapted to the domain:
- **Tag groups.** Danbooru groups only apply to booru-trained models (Anima, Illustrious, NoobAI, Pony). On Flux, Qwen-Image or Krea 2 sentences would be used.
- **WD tagger.** Anime only. For photorealism, measure with SigLIP or a VLM.
- **Prefix and negative.** The ones each model recommends.

## Dependent on Anima's architecture

| Anima detail | Observed consequence | What to expect on other models (untested) |
|---|---|---|
| Qwen3 0.6B → LLM adapter with the T5 ids as queries | The tag must be injected before the adapter; injected after it has almost no effect | Each model has its own injection point |
| Cross-attention keys with RMSNorm, no bias, and v_norm Identity | Weight only scales the values, so weight does not change where the image looks. That is why weight and attention are different mechanisms | Flux, SD3 and Qwen-Image also normalize q/k and probably behave the same. SDXL/SD1.5 have no QK-norm: weight also changes the keys |
| Separate cross-attention; context without positional encoding | Bias and normalization detect the cross-attention by key length, and the tag can go at the end of the prompt | In MMDiTs (Flux, SD3, Qwen-Image) text and image share attention and text tokens change in every block. Attention and weight_norm would need a redesign |
| T5 tokenizer (`fisheye` → `fish` + `eye`) | Forced, the tag is read literally (giant eye) | The pattern probably reappears, with other words depending on the tokenizer |
| Measured ranges (weight ≈ 4, weight_norm up to 16, attention 4–6) | These are this model's thresholds | They do not transfer; measure again |

Proposed contrast: a booru model on SDXL. Same tags, different architecture (CLIP, U-Net, no QK-norm), so it separates what is Anima-specific from what is general.

## Sources

1. Anima model card: https://huggingface.co/circlestone-labs/Anima/blob/main/README.md
2. ComfyUI_ADV_CLIP_emb (BlenderNeko), weight interpretation modes: https://runcomfy.com/comfyui-nodes/ComfyUI_ADV_CLIP_emb
3. Hertz et al., *Prompt-to-Prompt Image Editing with Cross Attention Control*, 2022. arXiv:2208.01626
4. Brack et al., *SEGA: Instructing Diffusion using Semantic Dimensions*, 2023. arXiv:2301.12247 — https://github.com/ml-research/semantic-image-editing
5. Lin et al., *Common Diffusion Noise Schedules and Sample Steps are Flawed*, 2023. arXiv:2305.08891 (RescaleCFG)
6. Sadat et al., *Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models* (APG), 2024. arXiv:2410.02416
7. Gandikota et al., *Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models*, ECCV 2024. arXiv:2311.12092 — https://sliders.baulab.info/
8. Ezra et al., *FreeSliders: Training-Free, Modality-Agnostic Concept Sliders*, 2025. arXiv:2511.00103
9. Sridhar and Vasconcelos, *Prompt Sliders for Fine-Grained Control, Editing and Erasing of Concepts in Diffusion Models*, ECCV-W 2024. arXiv:2409.16535
10. Baumann et al., *Continuous, Subject-Specific Attribute Control in T2I Models by Identifying Semantic Directions*, CVPR 2025. arXiv:2403.17064
11. Armandpour et al., *Re-imagine the Negative Prompt Algorithm* (Perp-Neg), 2023. arXiv:2304.04968 — https://perp-neg.github.io
12. AdaVD, concept erasure by orthogonal complement in the values. arXiv:2412.06143
13. Dalva et al., *FluxSpace: Disentangled Semantic Editing in Rectified Flow Models*, CVPR 2025. arXiv:2412.09611
14. Gandikota et al., *SliderSpace: Decomposing the Visual Capabilities of Diffusion Models*, ICCV 2025
15. Cao et al., *MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing*, ICCV 2023. arXiv:2304.08465

## Ideas for after the benchmark

- **Contrastive axis in guidance** (inspired by FreeSliders [8]). On `a | b` lines, push in the direction (with b) − (with a) instead of picking one tag by sign. What both poles share should cancel out and the slider would cross zero continuously. Cost: one extra pass per axis.
- **Projection to disentangle** (Perp-Neg [11], APG [6]). Remove from Δ_tag its component along the dragged concept's direction, e.g. Δ_monochrome for fisheye.
