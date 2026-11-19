#!/usr/bin/env python3
"""Aggregate a rocprofv3 --kernel-trace CSV into per-kernel / per-shape time tables.

rocprofiler-sdk serialises dispatch queues while recording kernel timestamps, so the
absolute wall span of the traced run is inflated -- but each dispatch's
End-Start delta is still its true execution time.  All shares below are therefore
computed over the SUM of kernel durations, not over the wall span.

Usage:
  i8_trace_top.py <run_kernel_trace.csv> [--hist] [--gap-us N] [--top N]
                                       [--seg last|all|N] [--min-dur-us N]
"""
import argparse
import collections
import csv
import re
import sys

RET_TYPES = ("void", "int", "float", "half", "_Float16", "bool", "unsigned", "double")


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("Kind") != "KERNEL_DISPATCH":
                continue
            try:
                s = int(r["Start_Timestamp"])
                e = int(r["End_Timestamp"])
            except (TypeError, ValueError):
                continue
            if e <= s:
                continue
            rows.append({
                "agent": r["Agent_Id"],
                "name": r["Kernel_Name"] or "",
                "s": s,
                "e": e,
                "dur": e - s,
                "gx": r.get("Grid_Size_X"),
                "gy": r.get("Grid_Size_Y"),
                "gz": r.get("Grid_Size_Z"),
                "wx": r.get("Workgroup_Size_X"),
                "wy": r.get("Workgroup_Size_Y"),
                "lds": r.get("LDS_Block_Size"),
                "vgpr": r.get("VGPR_Count"),
                "scr": r.get("Scratch_Size"),
            })
    return rows


def strip_ns(name):
    s = name
    s = s.replace("strata::kernels::(anonymous namespace)::", "")
    s = s.replace("strata::(anonymous namespace)::", "")
    s = s.replace("(anonymous namespace)::", "")
    s = s.replace("strata::kernels::", "")
    s = s.replace("strata::prefill::", "prefill::")
    s = s.replace("strata::", "")
    return s


def gemm_label(n):
    """Cijk_Alik_Bljk_HHS_BH_MT128x128x16_SE_... -> cijk MT128x128x16 SE"""
    m = re.search(r"_BH_(MT\d+x\d+x\d+)_(S[NE])_", n)
    lay = "TT" if "Alik_Bljk" in n else ("NT" if "Alik_Bljk" not in n else "?")
    if m:
        return f"cijk {m.group(1)} {m.group(2)}"
    return "cijk " + n[len("Cijk_"):][:40]


def bucket_name(name):
    if not name:
        return "<unnamed dispatch>"
    if name.startswith("Cijk_"):
        return gemm_label(name)
    s = strip_ns(name)
    if "(" in s:
        s = s.split("(", 1)[0]
    parts = s.split(" ")
    if len(parts) > 1 and parts[0] in RET_TYPES:
        s = " ".join(parts[1:])
    return s.strip()


def short(name, n=78):
    if len(name) <= n:
        return name
    return name[: n - 3] + "..."


def hist(rows, bucket_ms, top_agents=True):
    by_agent = collections.defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)
    for agent, ars in sorted(by_agent.items()):
        t0 = min(x["s"] for x in ars)
        buckets = collections.defaultdict(lambda: [0, 0])
        for x in ars:
            b = (x["s"] - t0) // (bucket_ms * 1_000_000)
            buckets[b][0] += 1
            buckets[b][1] += x["dur"]
        print(f"\n=== {agent}: per-{bucket_ms}ms bucket (n dispatches / busy ms)")
        line_n, line_b = [], []
        for b in sorted(buckets):
            n, d = buckets[b]
            line_n.append(f"{b}:{n}")
            line_b.append(f"{b}:{d/1e6:.0f}")
        print("  n   : " + "  ".join(line_n))
        print("  busy: " + "  ".join(line_b))


def segments(ars, gap_ns):
    out, cur, prev_e = [], [], None
    for r in sorted(ars, key=lambda x: x["s"]):
        if prev_e is not None and r["s"] - prev_e > gap_ns:
            if cur:
                out.append(cur)
            cur = []
        cur.append(r)
        prev_e = max(prev_e or 0, r["e"])
    if cur:
        out.append(cur)
    return out


def table(rows, keyfn, label, top, show_avg=True):
    agg = collections.OrderedDict()
    for r in rows:
        k = keyfn(r)
        a = agg.setdefault(k, {"n": 0, "dur": 0})
        a["n"] += 1
        a["dur"] += r["dur"]
    total = sum(a["dur"] for a in agg.values()) or 1
    items = sorted(agg.items(), key=lambda kv: -kv[1]["dur"])
    print(f"\n--- {label}: top {top} of {len(agg)} entries, kernel busy total {total/1e6:.1f} ms ---")
    hdr = f"{'#':>3} {'kernel / shape':<64} {'n':>6} {'avg us':>8} {'ms':>9} {'%':>6}"
    print(hdr)
    for i, (k, a) in enumerate(items[:top], 1):
        print(f"{i:>3} {short(k, 64):<64} {a['n']:>6} {a['dur']/a['n']/1e3:>8.1f} "
              f"{a['dur']/1e6:>9.2f} {100.0*a['dur']/total:>5.1f}%")
    rest = sum(a["dur"] for _, a in items[top:])
    if rest:
        print(f"{'':>3} {'(remaining)':<64} {'':>6} {'':>8} {rest/1e6:>9.2f} {100.0*rest/total:>5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--hist", action="store_true", help="print per-bucket dispatch density")
    ap.add_argument("--bucket-ms", type=int, default=1000)
    ap.add_argument("--gap-us", type=int, default=200000)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seg", default="last", help="last | all | <index>")
    ap.add_argument("--min-dur-us", type=float, default=0.0)
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        sys.exit("no KERNEL_DISPATCH rows")
    print(f"loaded {len(rows)} dispatches from {args.csv}")
    if args.min_dur_us > 0:
        rows = [r for r in rows if r["dur"] / 1e3 >= args.min_dur_us]

    if args.hist:
        hist(rows, args.bucket_ms)
        return

    by_agent = collections.defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)

    chosen = []
    for agent, ars in sorted(by_agent.items()):
        segs = segments(ars, args.gap_us * 1000)
        print(f"\n=== {agent}: {len(ars)} dispatches, {len(segs)} segments (gap>{args.gap_us}us)")
        for i, s in enumerate(segs):
            span = (max(x["e"] for x in s) - min(x["s"] for x in s)) / 1e6
            tot = sum(x["dur"] for x in s) / 1e6
            print(f"  seg{i:>2}: n={len(s):>6} span={span:>9.1f} ms busy={tot:>9.1f} ms "
                  f"({100.0*tot/span if span else 0:>5.1f}% busy) first={short(s[0]['name'], 44)}")
        if args.seg == "last":
            pick = segs[-1]
        elif args.seg == "all":
            pick = ars
        else:
            pick = segs[int(args.seg)]
        chosen.extend(pick)

    table(chosen, lambda r: bucket_name(r["name"]), "kernel family", args.top)
    table(chosen, lambda r: f"{bucket_name(r['name'])}  g={r['gx']}x{r['gy']}x{r['gz']} w={r['wx']}",
          "kernel + launch shape", args.top)
    table(chosen, lambda r: f"{bucket_name(r['name'])}  vgpr={r['vgpr']} lds={r['lds']} scr={r['scr']}",
          "kernel + resource footprint", args.top)


if __name__ == "__main__":
    main()
