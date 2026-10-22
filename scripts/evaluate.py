#!/usr/bin/env python3
"""
LabLens evaluation - one script, two suites, one results report.

  model     Base Qwen2.5-1.5B vs the fine-tuned adapter on the synthetic test sets.
            "test" = a lab layout never seen in training (generalisation);
            "val"  = layouts seen in training, different patients.
            Each model is scored on its raw output and after LabLens' code checks
            (unit conversion by code, made-up values rejected, unsupported tests removed).

  pipeline  The 8 files in data/test_files/ (digital PDFs, scanned PDF, phone photo,
            prescription) sent through the running API: file -> Docling -> gate -> model -> checks.

Usage (from the Lablens/ folder):
  python scripts/evaluate.py model                      # 50 reports per split, both models
  python scripts/evaluate.py model --n 20 --models finetuned
  python scripts/evaluate.py pipeline                   # needs: python -m uvicorn app.main:app --port 8000
  python scripts/evaluate.py all

Writes results/eval_<timestamp>.md (tables ready to paste into the README) and .json (every prediction).

Score = same metric as training (data/synth_data.py score_extraction): per expected test
0.6 value correct (±1%) + 0.2 unit + 0.2 flag, averaged, minus 0.5 x share of invented tests.
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))

from synth_data import score_extraction  # noqa: E402
from app.model import parse_json  # noqa: E402
from app.validate import validate  # noqa: E402

DATA = os.path.join(ROOT, "data")
SPLIT_NAMES = {"test": "unseen layout", "val": "seen layouts"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def field_stats(pred, gt):
    """Per-report counts: expected tests, found, value/unit/flag correct, invented."""
    want = {r["test"]: r for r in gt["results"]}
    got = {r["test"]: r for r in (pred or {}).get("results", []) if isinstance(r, dict) and "test" in r}
    s = dict(expected=len(want), found=0, value=0, unit=0, flag=0, predicted=len(got),
             invented=sum(1 for k in got if k not in want),
             date=int(isinstance(pred, dict) and pred.get("sample_date") == gt["sample_date"]))
    for k, g in want.items():
        r = got.get(k)
        if not r:
            continue
        s["found"] += 1
        try:
            s["value"] += abs(float(r.get("value")) - g["value"]) <= max(0.01 * abs(g["value"]), 0.011)
        except (TypeError, ValueError):
            pass
        s["unit"] += r.get("unit") == g["unit"]
        s["flag"] += r.get("flag") == g["flag"]
    return s


def strict_json(text):
    import re
    m = re.search(r"\{.*\}", text, re.S)
    try:
        json.loads(m.group(0)) if m else None
        return m is not None
    except json.JSONDecodeError:
        return False


def summarise(rows):
    n = len(rows)
    tot = {k: sum(r["fields"][k] for r in rows) for k in rows[0]["fields"]}
    found = max(tot["found"], 1)
    return {
        "n": n,
        "score": sum(r["score"] for r in rows) / n,
        "score_checked": sum(r["score_checked"] for r in rows) / n,
        "valid_json": sum(r["valid_json"] for r in rows) / n,
        "tests_found": tot["found"] / max(tot["expected"], 1),
        "value_acc": tot["value"] / found,
        "unit_acc": tot["unit"] / found,
        "flag_acc": tot["flag"] / found,
        "invented": tot["invented"] / max(tot["predicted"], 1),
        "date_acc": tot["date"] / n,
        "sec_per_report": sum(r["seconds"] for r in rows) / n,
    }


# ---------------------------------------------------------------------------
# Suite 1: model on the synthetic jsonl test sets
# ---------------------------------------------------------------------------
def load_model(adapter_dir, base_name):
    import torch
    import transformers
    from packaging.version import Version
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    tok.padding_side = "left"                      # needed for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    base = AutoModelForCausalLM.from_pretrained(base_name, **{kw: dtype})
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()   # not merged: base = adapter off
    return model, tok, device, torch


def generate_batch(model, tok, device, torch, prompts, max_new_tokens):
    texts = [tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(x, skip_special_tokens=True) for x in new], int((new != tok.pad_token_id).sum())


def run_model_suite(args):
    adapter = args.adapter or os.path.join(ROOT, "model_weights")
    print(f"Loading {args.base} + {adapter} ...")
    model, tok, device, torch = load_model(adapter, args.base)
    print(f"Running on {device}\n")

    results = {}
    for split in args.splits:
        path = os.path.join(DATA, f"extract_{split}.jsonl")
        if not os.path.exists(path):
            sys.exit(f"{path} not found. Generate the data first:\n"
                     "  python data/synth_data.py --patients 2000 --out data --seed 42")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.n]

        for which in args.models:
            label = f"{which} / {split}"
            out_rows = []
            t_split = time.perf_counter()
            for i in range(0, len(rows), args.batch):
                chunk = rows[i:i + args.batch]
                prompts = [[m for m in r["messages"] if m["role"] != "assistant"] for r in chunk]
                t0 = time.perf_counter()
                if which == "base":
                    with model.disable_adapter():
                        raws, _ = generate_batch(model, tok, device, torch, prompts, args.max_new_tokens)
                else:
                    raws, _ = generate_batch(model, tok, device, torch, prompts, args.max_new_tokens)
                per = (time.perf_counter() - t0) / len(chunk)
                for r, raw in zip(chunk, raws):
                    gt = json.loads(next(m["content"] for m in r["messages"] if m["role"] == "assistant"))
                    report = next(m["content"] for m in r["messages"] if m["role"] == "user")
                    pred = parse_json(raw)
                    checked, _ = validate(pred, report) if pred else (None, [])
                    out_rows.append(dict(
                        score=score_extraction(pred, gt) if pred else 0.0,
                        score_checked=score_extraction(checked, gt) if checked else 0.0,
                        valid_json=strict_json(raw), fields=field_stats(pred, gt),
                        seconds=per, raw=raw))
                done = min(i + args.batch, len(rows))
                avg = sum(x["score"] for x in out_rows) / len(out_rows)
                print(f"  {label:22s} {done:4d}/{len(rows)}  score so far {avg:.3f}  "
                      f"({time.perf_counter() - t_split:.0f}s)", end="\r", flush=True)
            print()
            results[label] = dict(model=which, split=split, summary=summarise(out_rows),
                                  predictions=[{k: v for k, v in x.items()} for x in out_rows])
    return results, device


def model_table(results, device):
    lines = ["| Model | Test set | n | Score | Score + code checks | Valid JSON | Tests found | "
             "Value | Unit | Flag | Invented | Date | s/report |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results.values():
        s = r["summary"]
        lines.append(
            f"| {'Fine-tuned (LoRA)' if r['model'] == 'finetuned' else 'Base Qwen2.5-1.5B'} "
            f"| {r['split']} ({SPLIT_NAMES.get(r['split'], '')}) | {s['n']} "
            f"| **{s['score']:.3f}** | {s['score_checked']:.3f} | {s['valid_json']:.0%} | {s['tests_found']:.0%} "
            f"| {s['value_acc']:.0%} | {s['unit_acc']:.0%} | {s['flag_acc']:.0%} | {s['invented']:.0%} "
            f"| {s['date_acc']:.0%} | {s['sec_per_report']:.1f} |")
    lines.append(f"\nDevice: {device}. Value/Unit/Flag accuracy are over the tests the model found. "
                 "Invented = share of predicted tests that are not in the report.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Suite 2: end-to-end pipeline on data/test_files via the running API
# ---------------------------------------------------------------------------
def run_pipeline_suite(args):
    import httpx

    folder = os.path.join(DATA, "test_files")
    truth = json.load(open(os.path.join(folder, "ground_truth.json"), encoding="utf-8"))
    try:
        health = httpx.get(f"{args.url}/health", timeout=10).json()
    except httpx.HTTPError:
        sys.exit(f"Can't reach {args.url}. Start the server first:\n  python -m uvicorn app.main:app --port 8000")
    print(f"API at {args.url}: model on {health.get('device')}, OCR engine {health.get('ocr_engine')}\n")

    rows = []
    for name, case in truth.items():
        print(f"  {name} ...", end="\r", flush=True)
        t0 = time.perf_counter()
        with open(os.path.join(folder, name), "rb") as f:
            r = httpx.post(f"{args.url}/analyze", params={"include_text": "false"},
                           files={"file": (name, f)}, timeout=900)
        wall = time.perf_counter() - t0
        j = r.json()
        row = dict(file=name, kind=case["kind"], about=case["about"], http=r.status_code, wall=wall,
                   route=(j.get("file") or {}).get("route"), response=j)
        if r.status_code != 200:
            row.update(score=0.0, note=f"HTTP {r.status_code}: {j.get('detail')}")
        else:
            pred = j.get("analysis")
            row.update(docling_s=j["extraction"]["seconds"], model_s=(j.get("model") or {}).get("seconds"))
            if case["kind"] == "negative":
                ok = pred is None or not pred.get("results")
                row.update(score=1.0 if ok else 0.0,
                           note="correctly rejected" if ok else "NOT rejected - model output shown")
            else:
                exp = case["expected"]
                row["score"] = score_extraction(pred, exp) if pred else 0.0
                fs = field_stats(pred, exp)
                corrected = sum(1 for x in (pred or {}).get("results", []) if "value_corrected_from" in x)
                note = f"{fs['found']}/{fs['expected']} tests"
                wrong = [k for k in ("value", "unit", "flag") if fs[k] < fs["found"]]
                if wrong:
                    note += ", errors in " + "/".join(wrong)
                if corrected:
                    note += f", {corrected} value(s) corrected by code"
                if (pred or {}).get("rejected_results"):
                    note += f", {len(pred['rejected_results'])} rejected as made up"
                row["note"] = note
        rows.append(row)
        print(f"  {name:42s} {row['score']:.2f}  {row['note']}")
    return rows, health


def pipeline_table(rows, health):
    lines = ["| File | What it tests | Route | Score | Docling s | Model s | Notes |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        ms = r.get("model_s")
        lines.append(f"| `{r['file']}` | {r['about']} | {r.get('route') or '-'} | **{r['score']:.2f}** "
                     f"| {r.get('docling_s', 0):.1f} | {'skipped' if ms is None else f'{ms:.1f}'} | {r['note']} |")
    mean = sum(r["score"] for r in rows) / len(rows)
    lines.append(f"\n**Mean: {mean:.3f}** over {len(rows)} files. Model on {health.get('device')}, "
                 f"OCR engine `{health.get('ocr_engine')}`.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", choices=["model", "pipeline", "all"])
    ap.add_argument("--n", type=int, default=50, help="reports per split (model suite)")
    ap.add_argument("--splits", default="test,val", help="comma list: test (unseen layout), val (seen layouts)")
    ap.add_argument("--models", default="finetuned,base", help="comma list: finetuned, base")
    ap.add_argument("--batch", type=int, default=4, help="reports generated together (lower if out of memory)")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="unsloth/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    args.splits = [s for s in args.splits.split(",") if s]
    args.models = [m for m in args.models.split(",") if m]

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    md = [f"# LabLens evaluation - {datetime.now():%Y-%m-%d %H:%M}\n"]
    dump = {"args": vars(args)}

    if args.suite in ("model", "all"):
        res, device = run_model_suite(args)
        md += ["## Model: extraction on synthetic reports\n", model_table(res, device), ""]
        dump["model"] = res
    if args.suite in ("pipeline", "all"):
        rows, health = run_pipeline_suite(args)
        md += ["## End-to-end pipeline: sample files through the API\n", pipeline_table(rows, health), ""]
        dump["pipeline"] = rows

    report = "\n".join(md)
    md_path = os.path.join(out_dir, f"eval_{stamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out_dir, f"eval_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=1, ensure_ascii=False, default=str)
    print("\n" + report)
    print(f"Saved: {md_path} (+ .json with every prediction)")


if __name__ == "__main__":
    main()
