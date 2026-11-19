#!/usr/bin/env python3
"""i8_bw.py -- effective HBM read bandwidth of the i8_kernel dispatches from a rocprofv3 counter csv.

usage: i8_bw.py <counter_collection.csv> [<kernel_trace.csv>]
The counter csv is long format (one row per counter per dispatch); the trace csv carries the grid dims.
Bytes per TCC_EA read request: gfx906 fetches a full 128-byte line per EA read request (the 32-byte
counter, TCC_EA_RDREQ_32B_sum, reads 0 on this part), which the weight bytes per dispatch confirm.
"""
import csv
import sys

LINE = 128.0

rows = list(csv.DictReader(open(sys.argv[1])))
disp = {}
for r in rows:
    k = r["Kernel_Name"]
    if "i8_kernel" not in k:
        continue
    d = r["Dispatch_Id"]
    a = disp.setdefault(d, {"k": k, "vgpr": r["VGPR_Count"], "scr": r["Scratch_Size"], "grid": r["Grid_Size"],
                            "t0": float(r["Start_Timestamp"]), "t1": float(r["End_Timestamp"]), "c": {}})
    a["c"][r["Counter_Name"]] = float(r["Counter_Value"] or 0)

by = {}
for a in disp.values():
    key = a["k"].split("i8_kernel")[1].split(">(")[0] + ">"
    b = by.setdefault(key, {"n": 0, "dur": 0.0, "rd": 0.0, "vgpr": [], "scr": []})
    b["n"] += 1
    b["dur"] += (a["t1"] - a["t0"]) / 1e9
    b["rd"] += a["c"].get("TCC_EA_RDREQ_sum", 0.0)
    b["vgpr"].append(int(a["vgpr"]))
    b["scr"].append(int(a["scr"]))

print(f"{'instantiation':<26} {'disp':>4} {'ms/disp':>8} {'EA rd req':>10} {'MB/disp':>8} {'GB/s':>7} {'VGPR':>5} {'scratch':>7}")
for k, b in by.items():
    per = b["dur"] / b["n"] * 1e3
    req = b["rd"] / b["n"]
    mb = req * LINE / 1e6
    print(f"{k:<26} {b['n']:>4} {per:>8.2f} {req:>10.0f} {mb:>8.2f} {mb/per:>7.1f} {max(b['vgpr']):>5} {max(b['scr']):>7}")

print(f"\n(requested line size assumed {LINE:.0f} B; peak MI50 HBM2 is 1024 GB/s, so the ratios above are 2-4% of peak)")
