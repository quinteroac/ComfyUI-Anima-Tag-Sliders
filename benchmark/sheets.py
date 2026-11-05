"""Contact sheets: one per tag (controls + full sweep) and one per extra group, saved as JPEG next to the phase folder.

python sheets.py --root /path/to/output/anima_tag_bench --phase phase1
"""
import argparse
import collections
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont

THUMB_W, THUMB_H, LABEL_H = 192, 281, 22
FONT = ImageFont.load_default(14)


def label(e):
    if "seed" in e:
        return f"s{e['seed']} " + label({k: v for k, v in e.items() if k != "seed"})
    if not e["sliders"]:
        return e["kind"]
    parts = []
    for line, mode, value, start, *options in e["sliders"]:
        tag = line.split("#")[0] if len(e["sliders"]) > 1 else ""
        # A = anchor, L0.35 = lock from 0.35, short enough to fit under the thumbnail.
        extra = "".join(f" {k[0].upper()}" + ("" if v is True else f"{v}") for o in options for k, v in o.items())
        parts.append(f"{tag} {mode} {value}" + (f" @{start}" if mode == "guidance" else "") + extra)
    return " + ".join(p.strip() for p in parts)


def sheet(root, entries, rows, title, path):
    cols = max(len(r) for r in rows)
    img = Image.new("RGB", (cols * THUMB_W, 30 + len(rows) * (THUMB_H + LABEL_H)), "white")
    draw = ImageDraw.Draw(img)
    draw.text((6, 6), title, fill="black", font=FONT)
    for y, row in enumerate(rows):
        for x, e in enumerate(row):
            if e is None:
                continue
            top = 30 + y * (THUMB_H + LABEL_H)
            img.paste(Image.open(os.path.join(root, e["file"])).convert("RGB").resize((THUMB_W, THUMB_H), Image.LANCZOS), (x * THUMB_W, top))
            draw.text((x * THUMB_W + 4, top + THUMB_H + 3), label(e)[:28], fill="black", font=FONT)
    img.save(path, quality=88)


def load(root, phase):
    with open(os.path.join(root, phase, "manifest.jsonl")) as f:
        return [e for e in map(json.loads, f) if e["status"] == "success"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--ref", nargs="*", default=[], help="phases whose none, written and same plain setting join each tag and seed of this phase")
    args = ap.parse_args()

    entries = load(args.root, args.phase)
    seeds_of = {(e["group"], e["tag"], e.get("seed", 42)) for e in entries}
    plain = collections.defaultdict(set)
    for e in entries:
        plain[(e["group"], e["tag"])] |= {tuple(s[:4]) for s in e["sliders"]}
    # Phase 1 entries predate the seed field: seed 42.
    refs = [{"seed": 42, **e} for p in args.ref for e in load(args.root, p) if (e["group"], e["tag"], e.get("seed", 42)) in seeds_of
            and (e["kind"] in ("none", "written") or (len(e["sliders"]) == 1 and tuple(e["sliders"][0]) in plain[(e["group"], e["tag"])]))]
    # Keep reference settings every seed has, so columns line up.
    have = collections.Counter((e["group"], e["tag"], e["kind"], json.dumps(e["sliders"])) for e in refs)
    seeds = collections.Counter((g, t) for g, t, _ in seeds_of)
    entries = [e for e in refs if have[(e["group"], e["tag"], e["kind"], json.dumps(e["sliders"]))] == seeds[(e["group"], e["tag"])]] + entries
    out = os.path.join(args.root, f"{args.phase}_sheets")
    os.makedirs(out, exist_ok=True)

    tags = collections.defaultdict(list)
    for e in entries:
        tags[(e["group"], e["tag"])].append(e)
    for (group, tag), es in tags.items():
        name = re.sub(r"[^a-z0-9]+", "_", f"{group}_{tag}".lower().replace("@", "at ")).strip("_")[:60]
        seeds = sorted({e.get("seed", 42) for e in es})
        if len(seeds) > 1:
            # One row per seed, same settings in the same columns: controls, then by option and strength.
            column = lambda e: (e["kind"] != "none", e["kind"] != "written", [f"{k}{v}" for s in e["sliders"] for o in s[4:] for k, v in o.items()], [s[2] for s in e["sliders"]])
            rows = [sorted((e for e in es if e["seed"] == s), key=column) for s in seeds]
        elif group.startswith("x_"):
            rows = [es[i:i + 8] for i in range(0, len(es), 8)]
        else:
            # Row 1: controls, weight; row 2: weight_norm, attention; row 3: guidance from 0, from 0.35.
            mode = lambda m, st=None: [e for e in es if e["sliders"] and e["sliders"][0][1] == m and (st is None or e["sliders"][0][3] == st)]
            rows = [[e for e in es if not e["sliders"]] + mode("weight"), mode("weight_norm") + mode("attention"), mode("guidance", 0.0) + mode("guidance", 0.35)]
        sheet(args.root, es, rows, f"{group} / {tag} / base {es[0]['base']} / {es[0]['unet'].split('/')[-1]}", os.path.join(out, f"{name}.jpg"))
    print(f"{len(tags)} sheets in {out}")


if __name__ == "__main__":
    main()
