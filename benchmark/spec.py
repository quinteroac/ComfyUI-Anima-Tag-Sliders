"""Tag slider benchmark jobs. Phase 1: one seed, every group x mode x strength. Phase 2: usable strengths on more seeds."""

SEED = 42
WIDTH, HEIGHT = 832, 1216
BASE_MODEL = dict(unet="anima/anima-base-v1.0.safetensors", steps=30, cfg=4.0)
TURBO_MODEL = dict(unet="anima/anima-turbo-v1.0-int8-convrot.safetensors", steps=10, cfg=1.0)
AESTHETIC_MODEL = dict(unet="anima/anima-aesthetic-v1.1.safetensors", steps=30, cfg=4.0)
NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"

PREFIX = "masterpiece, best quality, score_7, safe, "
BASES = {
    "P1": PREFIX + "1girl, solo, school uniform, standing, city street, full body",
    "P2": PREFIX + "1girl, solo, school uniform, portrait, upper body, looking at viewer, city street",
    "P3": PREFIX + "no humans, scenery, park, daytime",
    "PA": PREFIX + "1girl, solo, school uniform, city street, full body",
    "PQ": "safe, 1girl, solo, school uniform, standing, city street, full body",
    "P4": PREFIX + "A girl in a school uniform stands on a busy city street. The afternoon light falls softly across the tall buildings behind her.",
    "AE": "masterpiece, best quality, safe, 1girl, solo, school uniform, standing, city street, full body",
}

# (group, tag, base). Tags use Anima's format: spaces, score_N with underscore, artists with @.
TAGS = [
    ("composition", "from above", "P1"),
    ("composition", "dutch angle", "P1"),
    ("composition", "fisheye", "P1"),
    ("style", "watercolor (medium)", "P1"),
    ("style", "chibi", "P1"),
    ("style", "monochrome", "P1"),
    ("body", "big eyes", "P2"),
    ("body", "very long hair", "P1"),
    ("body", "red eyes", "P2"),
    ("clothes", "hat", "P1"),
    ("clothes", "kimono", "P1"),
    ("clothes", "thighhighs", "P1"),
    ("objects", "umbrella", "P1"),
    ("objects", "holding sword", "P1"),
    ("objects", "smartphone", "P1"),
    ("creatures", "cat", "P1"),
    ("creatures", "dragon", "P3"),
    ("creatures", "slime", "P1"),
    ("plants", "cherry blossoms", "P3"),
    ("plants", "sunflower", "P3"),
    ("plants", "ivy", "P3"),
    ("artists", "@kantoku", "P1"),
    ("artists", "@wlop", "P1"),
    ("artists", "@tora tsugumi", "P1"),
    ("artists", "kantoku", "P1"),
    ("characters", "hatsune miku", "P1"),
    ("characters", "frieren", "P1"),
    ("characters", "ijichi nijika", "P1"),
    ("quality", "masterpiece", "PQ"),
    ("quality", "score_9", "PQ"),
    ("quality", "worst quality", "PQ"),
    ("quality", "score_1", "PQ"),
    ("time", "year 2005", "P1"),
    ("time", "year 2024", "P1"),
    ("meta", "anime screenshot", "P1"),
    ("meta", "official art", "P1"),
    ("meta", "jpeg artifacts", "P1"),
    ("actions", "running", "PA"),
    ("actions", "eating", "PA"),
    ("actions", "jumping", "PA"),
    ("natural_language", "She is holding a red umbrella while it rains.", "P1"),
    ("natural_language", "from above", "P4"),
    ("natural_language", "watercolor (medium)", "P4"),
]

# (mode, strength, start_percent). start only matters for guidance.
SWEEP = (
    [("weight", s, 0.0) for s in (0.5, 2, 4, 8)]
    + [("weight_norm", s, 0.0) for s in (2, 4, 8, 16)]
    + [("attention", s, 0.0) for s in (2, 4, 8)]
    + [("guidance", s, start) for start in (0.0, 0.35) for s in (-2, 2, 4, 8)]
)


def job(group, tag, base, kind, model=BASE_MODEL, sliders=(), extra="", seed=SEED):
    """sliders: list of (line, mode, value, start) applied as chained AnimaTagSliders nodes."""
    return dict(group=group, tag=tag, base=base, kind=kind, model=model, sliders=list(sliders), extra=extra, seed=seed)


def slider(tag, mode, value, start=0.0, line=None, **options):
    """options: extra guidance inputs (anchor), stored as a fifth item only when given."""
    return (line or f"{tag}#64", mode, value, start) + ((options,) if options else ())


def jobs():
    out = []
    for group, tag, base in TAGS:
        out.append(job(group, tag, base, "none"))
        out.append(job(group, tag, base, "written", extra=tag))
        out += [job(group, tag, base, "sweep", sliders=[slider(tag, m, s, st)]) for m, s, st in SWEEP]

    # Axes and sequences.
    out += [job("x_axis", "from below | from above", "P1", "axis", sliders=[slider(None, "weight_norm", v, line="from below#16 | from above#16")]) for v in (-8, -4, 4, 8)]
    for mode, line in (("weight", "old | mid | recent | newest"), ("weight_norm", "old#3 | mid#3 | recent#3 | newest#3")):
        out += [job("x_sequence", line, "P1", "sequence", sliders=[slider(None, mode, p, line=line)]) for p in (1, 1.5, 2, 2.5, 3, 3.5, 4)]

    # Inverse push, deeper than the main sweep.
    for group, tag, base in (("quality", "worst quality", "PQ"), ("meta", "jpeg artifacts", "P1"), ("body", "big eyes", "P2"), ("time", "newest", "P1")):
        if tag == "newest":
            out.append(job("x_inverse", tag, base, "none"))
        out += [job("x_inverse", tag, base, "inverse", sliders=[slider(tag, "guidance", v, st)]) for st in (0.0, 0.35) for v in (-4, -8)]

    # Tag already in the prompt: suppress or reinforce it.
    out.append(job("x_existing", "smile", "P2", "none", extra="smile"))
    out += [job("x_existing", "smile", "P2", "existing", extra="smile", sliders=[slider("smile", "guidance", v, 0.35)]) for v in (-2, -4, -8)]
    out.append(job("x_existing", "smile", "P2", "existing", extra="smile", sliders=[slider("smile", "weight_norm", 4)]))

    # Two chained nodes with different modes.
    out.append(job("x_chain", "big eyes", "P1", "chain_single", sliders=[slider("big eyes", "guidance", 4, 0.35)]))
    out.append(job("x_chain", "fisheye + big eyes", "P1", "chain", sliders=[slider("fisheye", "weight_norm", 8), slider("big eyes", "guidance", 4, 0.35)]))

    # Other Anima variants: turbo (CFG 1) and aesthetic.
    for name, model, base in (("x_turbo", TURBO_MODEL, "P1"), ("x_aesthetic", AESTHETIC_MODEL, "AE")):
        for tag, setting in (("fisheye", slider("fisheye", "weight_norm", 8)), ("from above", slider("from above", "weight", 2)), ("hat", slider("hat", "guidance", 4, 0.35))):
            out.append(job(name, tag, base, "none", model=model))
            out.append(job(name, tag, base, "written", model=model, extra=tag))
            out.append(job(name, tag, base, "sweep", model=model, sliders=[setting]))

    for i, j in enumerate(out):
        j["id"] = f"{i:04d}"
    return out


# Phase 2: the usable strength of each mode from phase 1, on three more seeds (seed 42 is phase 1).
# none vs written across seeds also measures how much each tag moves the image compared to the seed.
PHASE2_SEEDS = (1, 2, 3)
PHASE2_SWEEP = [("weight", 2, 0.0), ("weight_norm", 4, 0.0), ("attention", 4, 0.0), ("guidance", 4, 0.35)]


def phase2_jobs():
    out = []
    for seed in PHASE2_SEEDS:
        for group, tag, base in TAGS:
            out.append(job(group, tag, base, "none", seed=seed))
            out.append(job(group, tag, base, "written", extra=tag, seed=seed))
            out += [job(group, tag, base, "sweep", sliders=[slider(tag, m, s, st)], seed=seed) for m, s, st in PHASE2_SWEEP]
        out.append(job("x_existing", "smile", "P2", "none", extra="smile", seed=seed))
        out += [job("x_existing", "smile", "P2", "existing", extra="smile", sliders=[slider("smile", "guidance", v, 0.35)], seed=seed) for v in (-4, -8)]
        out.append(job("x_inverse", "big eyes", "P2", "inverse", sliders=[slider("big eyes", "guidance", -4, 0.35)], seed=seed))
    for i, j in enumerate(out):
        j["id"] = f"{i:04d}"
    return out


def chain_jobs():
    """Re-run of the broken phase 1 chain after giving guidance's pass without the tag its own weight_norm arrays, in both node orders."""
    fisheye, big_eyes = slider("fisheye", "weight_norm", 8), slider("big eyes", "guidance", 4, 0.35)
    out = [job("x_chain", "fisheye + big eyes", "P1", "chain", sliders=[fisheye, big_eyes]),
           job("x_chain", "big eyes + fisheye", "P1", "chain", sliders=[big_eyes, fisheye]),
           job("x_chain", "fisheye", "P1", "chain_single", sliders=[fisheye]),
           # Same chain with big eyes at strength 1 (no push), and with big eyes written: is the gray from the push or from the tag?
           job("x_chain", "fisheye + big eyes", "P1", "chain", sliders=[fisheye, slider("big eyes", "guidance", 1, 0.35)]),
           job("x_chain", "fisheye", "P1", "chain_single", sliders=[fisheye], extra="big eyes"),
           job("x_chain", "fisheye + big eyes", "P1", "chain", sliders=[fisheye, slider("big eyes", "weight", 1)])]
    for i, j in enumerate(out):
        j["id"] = f"{i:04d}"
    return out


def chain_gray_jobs():
    """Why fisheye weight_norm 8 plus a second injected tag goes gray: second tag, strength, mode, order and seed."""
    fisheye = lambda m, s: slider("fisheye", m, s)
    second = lambda tag: slider(tag, "weight", 1)
    cases = [
        ("big eyes", [second("big eyes")]),
        ("fisheye + big eyes", [fisheye("weight", 1), second("big eyes")]),
        ("fisheye + big eyes", [fisheye("weight_norm", 4), second("big eyes")]),
        ("fisheye + big eyes", [fisheye("weight", 8), second("big eyes")]),
        ("fisheye + hat", [fisheye("weight_norm", 8), second("hat")]),
        ("fisheye + smile", [fisheye("weight_norm", 8), second("smile")]),
        ("big eyes + fisheye", [second("big eyes"), fisheye("weight_norm", 8)]),
    ]
    out = [job("x_chain_gray", tag, "P1", "chain", sliders=s) for tag, s in cases]
    for seed in PHASE2_SEEDS:
        out.append(job("x_chain_gray", "fisheye", "P1", "chain_single", sliders=[fisheye("weight_norm", 8)], seed=seed))
        out.append(job("x_chain_gray", "fisheye + big eyes", "P1", "chain", sliders=[fisheye("weight_norm", 8), second("big eyes")], seed=seed))
    for i, j in enumerate(out):
        j["id"] = f"{i:04d}"
    return out


# Freezing the image: guidance from 0.35 anchored to the prompt without the tag, and with the self-attention lock
# (from 0.35 at strength 4, and from 0.55 at strength 8 to leave room for shape changes), on local tags and 4 seeds.
# none, written and plain guidance 4 from 0.35 for these tags and seeds come from phase 1 and phase 2.
# The lock gave nothing a lower anchor strength did not and was removed from the node afterwards, so its jobs no longer rerun.
LOCK_TAGS = [t for t in TAGS if t[1] in ("big eyes", "hat", "kimono", "cat", "dragon", "smartphone")]
# Each later sweep was added after a run, so earlier ids stay put: anchor and the late lock at the same strengths,
# then anchor 6, where the anchor curve meets lock 8 from 0.55.
LOCK_SWEEPS = [[(4, dict(anchor=True)), (4, dict(lock=0.35)), (8, dict(lock=0.55))],
               [(8, dict(anchor=True)), (4, dict(lock=0.55))],
               [(6, dict(anchor=True))]]


def lock_jobs():
    out = []
    for sweep in LOCK_SWEEPS:
        for seed in (SEED,) + PHASE2_SEEDS:
            for group, tag, base in LOCK_TAGS:
                out += [job(group, tag, base, "sweep", sliders=[slider(tag, "guidance", s, 0.35, **o)], seed=seed) for s, o in sweep]
    for i, j in enumerate(out):
        j["id"] = f"{i:04d}"
    return out


PHASES = {"phase1": jobs, "phase2": phase2_jobs, "chain_fix": chain_jobs, "chain_gray": chain_gray_jobs, "lock": lock_jobs}
