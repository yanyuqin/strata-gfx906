#!/usr/bin/env python3
"""mi50_bench.py -- non-streaming OpenAI client for the MI50 Q4_K_XL A/B.

Same prompt text for every arm at a given (case, rep): the filler is fixed and only the nonce
changes, so prompt_n matches across versions and nothing hits the prefix cache.  Metrics come
from the engine's own `timings` block (prompt_ms / prompt_per_second, predicted_ms /
predicted_per_second, draft_n / draft_n_accepted), i.e. the numbers the engine measured,
not a client-side guess.

  python3 mi50_bench.py --port 8010 --case 5k --reps 3 --gen 128
"""
import argparse, json, statistics, sys, time, urllib.request

FILLER = (
    "The engine streams sparse experts from host memory to the card while it decodes, "
    "and the scheduler overlaps the next window with the verification of the current one. "
    "A prompt that is long enough exercises the prefill path, where the dense products and "
    "the expert stage run over the whole sequence before the first token appears. "
)

CASES = {"short": 512, "2k": 2000, "5k": 5300, "mid": 9500, "21k": 21000, "32k": 32000}


def build_prompt(target_tokens, nonce):
    words = max(16, int(target_tokens / 1.33))
    buf = ["[%s] " % nonce]
    while len(" ".join(buf).split()) < words:
        buf.append(FILLER)
    return " ".join(buf) + "\n\nReply with one short sentence summarising the text above."


def model_id(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as r:
        d = json.loads(r.read())
    return d["data"][0]["id"]


def post(port, model, prompt, gen, timeout=1800):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": gen, "seed": 42,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--case", default="5k")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--warmup", action="store_true", default=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    target = CASES[a.case] if a.case in CASES else int(a.case)
    mid = model_id(a.port)
    rows = []
    for rep in range(a.reps + (1 if a.warmup else 0)):
        nonce = f"mi50-{a.case}-{rep}"
        d, wall = post(a.port, mid, build_prompt(target, nonce), a.gen)
        t = d.get("timings") or {}
        u = d.get("usage") or {}
        row = {
            "rep": rep, "warmup": rep == 0 and a.warmup, "case": a.case,
            "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "completion_n": t.get("predicted_n"),
            "prefill_ms": t.get("prompt_ms"),
            "prefill_tps": t.get("prompt_per_second"),
            "decode_tps": t.get("predicted_per_second"),
            "ttft_s": round(t.get("prompt_ms", 0) / 1000.0, 3),
            "wall_s": round(wall, 3),
            "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
        }
        if not row["warmup"]:
            rows.append(row)
        print(json.dumps(row), flush=True)

    def med(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(statistics.median(vals), 3) if vals else None

    out = {"tag": a.tag, "case": a.case, "reps": a.reps, "gen": a.gen,
           "prompt_n": rows[0]["prompt_n"] if rows else None,
           "prefill_tps": med("prefill_tps"), "decode_tps": med("decode_tps"),
           "ttft_s": med("ttft_s"), "prefill_ms": med("prefill_ms"), "wall_s": med("wall_s"),
           "draft_n": med("draft_n"), "draft_acc": med("draft_acc"), "rows": rows}
    print("SUMMARY " + json.dumps(out), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    sys.exit(main())
