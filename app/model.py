"""
Step 3: run the fine-tuned LabLens model (Qwen2.5-1.5B + LoRA adapter) on the extracted text.

Loaded once at server start-up. Long reports are split into chunks that fit the
2048-token length the model was trained on, and the per-chunk results are merged.
"""
import json
import logging
import re
import threading
import time

from . import config
from .prompts import SYSTEM_EXTRACT

log = logging.getLogger("lablens.model")


class LabModel:
    def __init__(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():          # Apple Silicon GPU
            self.device, dtype = "mps", torch.float16
        else:
            self.device, dtype = "cpu", torch.float32
        log.info("Loading %s + adapter %s on %s", config.BASE_MODEL, config.ADAPTER_DIR, self.device)

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(config.ADAPTER_DIR))
        import transformers
        from packaging.version import Version
        dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, **{dtype_kw: dtype})
        model = PeftModel.from_pretrained(base, str(config.ADAPTER_DIR)).merge_and_unload()
        self.model = model.to(self.device).eval()
        self._lock = threading.Lock()        # one generation at a time (GPU memory / MPS safety)

    # -- helpers ---------------------------------------------------------------
    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def chunk(self, text: str, budget: int) -> list[str]:
        """Split on blank lines (Docling's block separator) so tables/sections stay whole."""
        if self.n_tokens(text) <= budget:
            return [text]
        chunks, cur = [], []
        for block in re.split(r"\n\s*\n", text):
            candidate = "\n\n".join(cur + [block])
            if cur and self.n_tokens(candidate) > budget:
                chunks.append("\n\n".join(cur))
                cur = [block]
            else:
                cur.append(block)
        if cur:
            chunks.append("\n\n".join(cur))
        return chunks

    def generate(self, report_text: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_EXTRACT},
                    {"role": "user", "content": report_text}]
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(self.device)
        with self._lock, self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # -- main entry ------------------------------------------------------------
    def analyze(self, report_text: str) -> dict:
        t0 = time.perf_counter()
        chunks = self.chunk(report_text, config.CHUNK_TOKENS)
        raws, parsed = [], []
        for c in chunks:
            raw = self.generate(c)
            raws.append(raw)
            parsed.append(parse_json(raw))
        merged = merge(parsed)
        problems = merged is None or merged.get("_salvaged") or any(p is None for p in parsed)
        return {"result": merged, "chunks": len(chunks), "device": self.device,
                "raw_outputs": raws if problems else None,
                "seconds": round(time.perf_counter() - t0, 2)}


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return salvage_json(text)


def _loads_lenient(s: str):
    s = re.sub(r"(?<=[:\s\[,])0+(\d)", r"\1", s)         # 08.1 -> 8.1 (leading zeros aren't valid JSON)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def salvage_json(text: str):
    """Recover what we can from cut-off or slightly broken output: every complete result object."""
    results = []
    tail = text.split('"results"', 1)[-1] if '"results"' in text else ""
    for obj in re.findall(r"\{[^{}]*\}", tail):
        r = _loads_lenient(obj)
        if isinstance(r, dict) and "test" in r:
            results.append(r)
    if not results:
        return None
    out = {"sample_date": None, "patient": None, "results": results, "_salvaged": True}
    d = re.search(r'"sample_date"\s*:\s*"([^"]*)"', text)
    if d:
        out["sample_date"] = d.group(1)
    p = re.search(r'"patient"\s*:\s*(\{[^{}]*\})', text)
    if p:
        out["patient"] = _loads_lenient(p.group(1))
    return out


def merge(parts: list) -> dict | None:
    """Combine per-chunk outputs: first non-empty date/patient, union of results (first wins)."""
    parts = [p for p in parts if isinstance(p, dict)]
    if not parts:
        return None
    out = {"sample_date": None, "patient": None, "results": []}
    if any(p.get("_salvaged") for p in parts):
        out["_salvaged"] = True
    seen = set()
    for p in parts:
        out["sample_date"] = out["sample_date"] or p.get("sample_date")
        if not out["patient"] and isinstance(p.get("patient"), dict) and any(p["patient"].values()):
            out["patient"] = p["patient"]
        for r in p.get("results") or []:
            key = r.get("test") if isinstance(r, dict) else None
            if key and key not in seen:
                seen.add(key)
                out["results"].append(r)
    return out
