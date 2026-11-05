"""Aggregates metrics.jsonl into the numbers used by the report.

python analyze.py --root /path/to/output/anima_tag_bench

Phase 1, per mode and strength (median/mean over the 43 tags, seed 42):
  adh_siglip  (setting - none) / (written - none) of the SigLIP text cosine; 1 = like writing the tag
  adh_wd      same with the WD probability, only for tags in the tagger's vocabulary
  change      1 - cosine of SigLIP image embeddings against the image without the tag
  base, gray, color, hf  WD base-prompt tags, WD monochrome/greyscale, colorfulness, |Laplacian| * 1000
Phase 2 adds seeds 1-3 to seed 42:
  text vs seed  change none -> written, against the change between two seeds without the tag
  per mode      adh_siglip and change per seed, as mean and spread
The lock phase compares guidance from 0.35 with anchor and the self-attention lock on 6 local tags and the same 4 seeds:
  area          share of pixels that visibly differ from the image without the tag; lower = more of the image frozen
"""
import argparse
import collections
import itertools
import json
import os

import numpy as np
from PIL import Image

MAIN = ("none", "written", "sweep")


def load(root, phase):
    folder = os.path.join(root, phase)
    if not os.path.exists(os.path.join(folder, "metrics.jsonl")):
        return []
    with open(os.path.join(folder, "manifest.jsonl")) as f:
        manifest = {e["id"]: e for e in map(json.loads, f)}
    rows = []
    with open(os.path.join(folder, "metrics.jsonl")) as f:
        for m in map(json.loads, f):
            e = {**manifest[m["id"]], **m}
            e["seed"] = e.get("seed", 42)
            e["embedding"] = np.array(e["embedding"])
            e["key"] = setting(e)
            rows.append(e)
    return rows


def setting(e):
    if not e["sliders"]:
        return e["kind"]
    _, mode, value, start, *options = e["sliders"][0]
    return f"{mode} {value}" + (f" @{start}" if mode == "guidance" else "") + "".join(f" {k} {v}" for o in options for k, v in o.items())


def change(a, b):
    return 1.0 - float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def gain(x, none, written, field, floor):
    if x[field] is None or written[field] - none[field] < floor:
        return None
    return (x[field] - none[field]) / (written[field] - none[field])


def by_tag_and_seed(rows):
    out = collections.defaultdict(dict)
    for r in rows:
        if not r["group"].startswith("x_") and r["kind"] in MAIN:
            out[(r["group"], r["tag"], r["seed"])][r["key"]] = r
    return out


def phase1_modes(rows):
    table = by_tag_and_seed(rows)
    keys = sorted({k for s in table.values() for k in s if k not in ("none", "written")})
    print(f"{'setting':22s} {'adh_siglip':>10s} {'adh_wd':>7s} {'change':>7s} {'base':>5s} {'gray':>5s} {'color':>6s} {'hf':>6s}")
    for k in ["written"] + keys:
        vals = collections.defaultdict(list)
        for s in table.values():
            if k not in s:
                continue
            x, none, written = s[k], s["none"], s["written"]
            for name, v in (("adh_siglip", gain(x, none, written, "siglip_text", 0.01)), ("adh_wd", gain(x, none, written, "wd_tag", 0.1))):
                if v is not None:
                    vals[name].append(v)
            vals["change"].append(change(x["embedding"], none["embedding"]))
            vals["base"].append(x["wd_base"] or 0.0)
            vals["gray"].append(x["wd_mono"])
            vals["color"].append(x["colorfulness"])
            vals["hf"].append(x["hf_energy"] * 1000)
        print(f"{k:22s} {np.median(vals['adh_siglip']):10.2f} {np.median(vals['adh_wd']) if vals['adh_wd'] else float('nan'):7.2f} "
              f"{np.mean(vals['change']):7.3f} {np.mean(vals['base']):5.2f} {np.mean(vals['gray']):5.2f} {np.mean(vals['color']):6.1f} {np.mean(vals['hf']):6.1f}")


def text_vs_seed(rows):
    table = by_tag_and_seed(rows)
    tags = sorted({(g, t) for g, t, _ in table})
    per_group = collections.defaultdict(list)
    print(f"\n{'group':18s} {'tag':48s} {'text':>6s} {'seed':>6s} {'ratio':>6s}")
    for g, t in tags:
        seeds = sorted(s for gg, tt, s in table if (gg, tt) == (g, t))
        text = [change(table[(g, t, s)]["written"]["embedding"], table[(g, t, s)]["none"]["embedding"]) for s in seeds]
        seed = [change(table[(g, t, a)]["none"]["embedding"], table[(g, t, b)]["none"]["embedding"]) for a, b in itertools.combinations(seeds, 2)]
        if not seed:
            continue
        ratio = np.mean(text) / np.mean(seed)
        per_group[g].append(ratio)
        print(f"{g:18s} {t[:48]:48s} {np.mean(text):6.3f} {np.mean(seed):6.3f} {ratio:6.2f}")
    print("\ngroup mean ratio (text change / seed change)")
    for g, v in sorted(per_group.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {g:18s} {np.mean(v):5.2f}")


def modes_across_seeds(rows):
    table = by_tag_and_seed(rows)
    phase2_keys = sorted({k for (g, t, seed), s in table.items() if seed != 42 for k in s if k not in ("none", "written")})
    seeds = sorted({seed for _, _, seed in table})
    print(f"\n{'setting':22s} " + " ".join(f"{'adh s' + str(s):>9s}" for s in seeds) + f" {'spread':>7s} {'change':>7s} {'gray':>5s}")
    for k in phase2_keys:
        adh_by_seed, ch, gray = [], [], []
        for seed in seeds:
            vals = []
            for (g, t, s), d in table.items():
                if s == seed and k in d:
                    v = gain(d[k], d["none"], d["written"], "siglip_text", 0.01)
                    if v is not None:
                        vals.append(v)
                    ch.append(change(d[k]["embedding"], d["none"]["embedding"]))
                    gray.append(d[k]["wd_mono"])
            adh_by_seed.append(np.median(vals))
        print(f"{k:22s} " + " ".join(f"{a:9.2f}" for a in adh_by_seed) + f" {np.std(adh_by_seed):7.2f} {np.mean(ch):7.3f} {np.mean(gray):5.2f}")


def changed_area(root, a, b):
    """Share of pixels that visibly differ between two images, at 1/4 size."""
    pixels = lambda e: np.asarray(Image.open(os.path.join(root, e["file"])).convert("RGB").reduce(4), dtype=np.float32) / 255
    return float((np.abs(pixels(a) - pixels(b)).max(axis=-1) > 0.15).mean())


def lock_compare(rows, lock, root):
    table = by_tag_and_seed(rows + lock)
    tags = sorted({(r["group"], r["tag"]) for r in lock})
    keys = ["written", "guidance 4 @0.35"] + sorted({r["key"] for r in lock})
    area = collections.defaultdict(dict)
    print(f"\n{'setting':40s} {'adh_siglip':>10s} {'adh_wd':>7s} {'change':>7s} {'area':>6s} {'gray':>5s}")
    for k in keys:
        adh, adh_wd, ch, ar, gray = [], [], [], [], []
        for (g, t, s), d in table.items():
            if (g, t) not in tags or k not in d:
                continue
            x, none, written = d[k], d["none"], d["written"]
            for out, v in ((adh, gain(x, none, written, "siglip_text", 0.01)), (adh_wd, gain(x, none, written, "wd_tag", 0.1))):
                if v is not None:
                    out.append(v)
            ch.append(change(x["embedding"], none["embedding"]))
            ar.append(changed_area(root, x, none))
            area[t].setdefault(k, []).append(ar[-1])
            gray.append(x["wd_mono"])
        print(f"{k:40s} {np.median(adh):10.2f} {np.median(adh_wd) if adh_wd else float('nan'):7.2f} {np.mean(ch):7.3f} {np.mean(ar):6.2f} {np.mean(gray):5.2f}")
    print("\nchanged area per tag (share of pixels that differ from the image without the tag)")
    short = lambda k: k.replace("guidance ", "g").replace(" @0.35", "").replace(" anchor True", " A").replace(" lock ", " L")
    print(f"{'tag':12s} " + " ".join(f"{short(k):>10s}" for k in keys))
    for t, by_key in sorted(area.items()):
        print(f"{t:12s} " + " ".join(f"{np.mean(by_key[k]):10.2f}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    phase1, phase2, lock = load(args.root, "phase1"), load(args.root, "phase2"), load(args.root, "lock")
    phase1_modes(phase1)
    if phase2:
        text_vs_seed(phase1 + phase2)
        modes_across_seeds(phase1 + phase2)
    if lock:
        lock_compare(phase1 + phase2, lock, args.root)


if __name__ == "__main__":
    main()
