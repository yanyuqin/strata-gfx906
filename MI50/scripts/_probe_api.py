import json, urllib.request

def post(port, prompt, gen=64):
    body = json.dumps({"model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": gen,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

d = post(8010, "Explain in three sentences why experts are streamed from host memory.")
print("keys:", list(d.keys()))
print("usage:", d.get("usage"))
print("timings:", json.dumps(d.get("timings"))[:500])
print("content:", (d["choices"][0]["message"].get("content") or "")[:120])
