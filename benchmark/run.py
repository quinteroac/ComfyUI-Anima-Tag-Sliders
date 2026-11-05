"""Queue the benchmark jobs on a running ComfyUI and log results to a manifest. Resumable: finished ids are skipped.

python run.py --url http://127.0.0.1:8189 --root /path/to/comfy/output/anima_tag_bench --phase phase1
"""
import argparse
import json
import os
import time
import urllib.request

import spec


def workflow(j, prefix):
    m = j["model"]
    text = spec.BASES[j["base"]] + (", " + j["extra"] if j["extra"] else "")
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": text}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": spec.NEGATIVE}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": spec.WIDTH, "height": spec.HEIGHT, "batch_size": 1}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }
    model = ["1", 0]
    for i, (line, mode, value, start, *options) in enumerate(j["sliders"]):
        inputs = {"model": model, "clip": ["2", 0], "tags": line, "slider_values": json.dumps({line: value}), "mode": mode, "bypass": False}
        if mode == "guidance":
            inputs.update({"mode.start_percent": start, "mode.end_percent": 1.0, "mode.rescale": 0.0, "mode.anchor": False})
            inputs.update({f"mode.{k}": v for o in options for k, v in o.items()})
        w[f"s{i}"] = {"class_type": "AnimaTagSliders", "inputs": inputs}
        model = [f"s{i}", 0]
    w["8"] = {"class_type": "KSampler", "inputs": {"model": model, "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
                                                   "seed": j["seed"], "steps": m["steps"], "cfg": m["cfg"], "sampler_name": "er_sde",
                                                   "scheduler": "simple", "denoise": 1.0}}
    return w, text


def call(url, path, data=None):
    req = urllib.request.Request(url + path, data=json.dumps(data).encode() if data else None, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8189")
    ap.add_argument("--root", required=True)
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", help="run only these job ids")
    args = ap.parse_args()

    manifest = os.path.join(args.root, args.phase, "manifest.jsonl")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    done = set()
    if os.path.exists(manifest):
        with open(manifest) as f:
            done = {e["id"] for e in map(json.loads, f) if e["status"] == "success"}
    todo = [j for j in spec.PHASES[args.phase]() if j["id"] not in done and (not args.ids or j["id"] in args.ids)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)

    for n, j in enumerate(todo, 1):
        w, text = workflow(j, f"{args.phase}/{j['id']}")
        pid = call(args.url, "/prompt", {"prompt": w})["prompt_id"]
        while True:
            hist = call(args.url, f"/history/{pid}")
            if pid in hist:
                break
            time.sleep(2)
        h = hist[pid]
        msgs = {m[0]: m[1] for m in h["status"].get("messages", [])}
        start = msgs.get("execution_start", {}).get("timestamp")
        end = (msgs.get("execution_success") or msgs.get("execution_error") or {}).get("timestamp")
        files = [im["filename"] for o in h["outputs"].values() for im in o.get("images", [])]
        entry = {**{k: v for k, v in j.items() if k != "model"}, "unet": j["model"]["unet"], "steps": j["model"]["steps"], "cfg": j["model"]["cfg"],
                 "prompt": text, "status": h["status"]["status_str"], "seconds": (end - start) / 1000 if start and end else None,
                 "file": f"{args.phase}/{files[0]}" if files else None,
                 "error": msgs.get("execution_error", {}).get("exception_message")}
        with open(manifest, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[{n}/{len(todo)}] {j['id']} {j['group']} {j['tag']} {j['kind']} {j['sliders']} -> {entry['status']} {entry['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
