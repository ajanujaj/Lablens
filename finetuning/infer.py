"""
LabLens inference: base Qwen2.5-1.5B-Instruct + your LoRA adapter (model_weights/).

Usage (run from the Lablens/ folder):
  python finetuning/infer.py data/samples/apex.txt            # one report file -> JSON
  python finetuning/infer.py --test 5                          # quick score (full eval: scripts/evaluate.py)
  python finetuning/infer.py --task drift --test 5             # if you train the drift adapter
"""
import argparse, json, os, re, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "unsloth/Qwen2.5-1.5B-Instruct"   # same base as training


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():          # Apple Silicon GPU
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(adapter_dir, merge=True):
    device, dtype = pick_device()
    print(f"Loading on {device} ({dtype}) ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(adapter_dir)   # tokenizer + chat template saved with the adapter
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter_dir)
    if merge:
        model = model.merge_and_unload()               # folds LoRA into base -> faster generation
    return model.to(device).eval(), tok, device


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def system_prompt(task):
    # Exact system prompt the model was trained with.
    if task == "extract":
        sys.path.insert(0, ROOT)
        from app.prompts import SYSTEM_EXTRACT
        return SYSTEM_EXTRACT
    row = load_jsonl(os.path.join(ROOT, "data", f"{task}_test.jsonl"))[0]
    return next(m["content"] for m in row["messages"] if m["role"] == "system")


@torch.inference_mode()
def generate(model, tok, device, messages, max_new_tokens=1024):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def score_extract(pred, gt):
    if not isinstance(pred, dict) or not isinstance(pred.get("results"), list):
        return 0.0
    want = {r["test"]: r for r in gt["results"]}
    got = {r["test"]: r for r in pred["results"] if isinstance(r, dict) and "test" in r}
    s = 0.0
    for k, g in want.items():
        r = got.get(k)
        if not r:
            continue
        try:
            ok_v = abs(float(r.get("value")) - g["value"]) <= max(0.01 * abs(g["value"]), 0.011)
        except (TypeError, ValueError):
            ok_v = False
        s += 0.6 * ok_v + 0.2 * (r.get("unit") == g["unit"]) + 0.2 * (r.get("flag") == g["flag"])
    invented = sum(1 for k in got if k not in want) / max(len(got), 1)
    return max(0.0, s / len(want) - 0.5 * invented)


def score_drift(pred, gt):
    if not isinstance(pred, dict) or not isinstance(pred.get("trends"), list):
        return 0.0
    got = {t.get("test"): t for t in pred["trends"] if isinstance(t, dict)}
    s = sum(0.5 * (got.get(g["test"], {}).get("direction") == g["direction"]) +
            0.5 * (got.get(g["test"], {}).get("in_range_drift") == g["in_range_drift"])
            for g in gt["trends"])
    return s / len(gt["trends"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", help="path to a lab report .txt file")
    ap.add_argument("--task", default="extract", choices=["extract", "drift"])
    ap.add_argument("--adapter", default=os.path.join(ROOT, "model_weights"))
    ap.add_argument("--test", type=int, default=0, help="evaluate on N test examples")
    args = ap.parse_args()

    model, tok, device = load_model(args.adapter)

    if args.test:
        rows = load_jsonl(os.path.join(ROOT, "data", f"{args.task}_test.jsonl"))
        random.seed(0)
        rows = random.sample(rows, min(args.test, len(rows)))
        score_fn = score_extract if args.task == "extract" else score_drift
        scores = []
        for i, r in enumerate(rows):
            prompt = [m for m in r["messages"] if m["role"] != "assistant"]
            gt = json.loads(next(m["content"] for m in r["messages"] if m["role"] == "assistant"))
            pred = parse_json(generate(model, tok, device, prompt))
            scores.append(score_fn(pred, gt))
            print(f"[{i+1}/{len(rows)}] score={scores[-1]:.2f}  valid_json={pred is not None}")
        print(f"\nMean score: {sum(scores)/len(scores):.3f}")
        return

    if not args.report:
        ap.error("give a report file, or use --test N")
    report = open(args.report, encoding="utf-8").read()
    messages = [{"role": "system", "content": system_prompt(args.task)},
                {"role": "user", "content": report}]
    raw = generate(model, tok, device, messages)
    parsed = parse_json(raw)
    print(json.dumps(parsed, indent=2, ensure_ascii=False) if parsed else raw)


if __name__ == "__main__":
    main()
