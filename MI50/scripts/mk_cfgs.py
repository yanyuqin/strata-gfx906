#!/usr/bin/env python3
"""mk_cfgs.py -- build the MI50 A/B configs from the production cfg (args unchanged).

  python3 mk_cfgs.py base        -> cfg-base.json    (v0.2.10 binary, ~/strata/build-906-0210)
  python3 mk_cfgs.py new         -> cfg-0214.json    (v0.2.14.1 gfx906 binary)
  python3 mk_cfgs.py arm NAME k=v ...   -> cfg-arm-NAME.json, args patched (k is an engine flag)
"""
import json, sys, os

SRC = "/home/yanyq/bench/cfg-ud31-q4-256k.json"
OUT = "/home/yanyq/bench/mi50-0214"
EXES = {
    "base": "/home/yanyq/strata/build-906-0210/strata",
    "new": "/home/yanyq/port/mi50-0214/build-906-0214/strata",
}
LOGS = {
    "base": "/home/yanyq/bench/mi50-0214/logs/engine-base.log",
    "new": "/home/yanyq/bench/mi50-0214/logs/engine-0214.log",
}


def patch_args(args, pairs):
    """pairs like prefill=32768, pcie-frac=0.55, spec=6 -- replace the flag's value or append it."""
    out = list(args)
    for p in pairs:
        if "=" not in p:
            raise SystemExit("bad override %r (need flag=value)" % p)
        k, v = p.split("=", 1)
        flag = k if k.startswith("--") else "--" + k
        if flag in out:
            out[out.index(flag) + 1] = v
        elif v == "DROP":          # remove the flag and its value
            if flag in out:
                i = out.index(flag)
                del out[i:i + 2]
        else:
            out += [flag, v]
    # DROP pass
    res = []
    i = 0
    while i < len(out):
        if out[i].startswith("--") and i + 1 < len(out) and out[i + 1] == "DROP":
            i += 2
            continue
        res.append(out[i])
        i += 1
    return res


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "new"
    d = json.load(open(SRC))
    os.makedirs(OUT + "/logs", exist_ok=True)
    if what in ("base", "new"):
        d["exe"] = EXES[what]
        d["log"] = LOGS[what]
        path = f"{OUT}/cfg-{what}.json"
    else:  # arm <name> flag=value...
        name = sys.argv[2]
        d["exe"] = EXES["new"]
        d["log"] = f"{OUT}/logs/engine-arm-{name}.log"
        d["args"] = patch_args(d["args"], sys.argv[3:])
        path = f"{OUT}/cfg-arm-{name}.json"
    json.dump(d, open(path, "w"), indent=1)
    print(path, "args=", len(d["args"]))
    print(" ".join(d["args"]))


if __name__ == "__main__":
    main()
