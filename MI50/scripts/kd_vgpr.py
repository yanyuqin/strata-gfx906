#!/usr/bin/env python3
# VGPR/SGPR/spill/LDS of each kernel from the amdhsa metadata in a gfx906 object's .note section.  Each kernel's record is a
# run of msgpack key/value pairs that starts at ".symbol" (the mangled name + ".kd"); the counts follow it.
import sys

data = open(sys.argv[1], "rb").read()
want = sys.argv[2] if len(sys.argv) > 2 else "i8_kernelILi"


def val(b, off):
    t = b[off]
    if t <= 0x7F:
        return t
    if t >= 0xE0:
        return t - 0x100
    if t >= 0xA0:
        return b[off + 1:off + 1 + (t & 0x1F)].decode("utf-8", "replace")
    if t == 0xC2:
        return False
    if t == 0xC3:
        return True
    return None


def at(b, key):
    j = b.find(key)
    return None if j < 0 else val(b, j + len(key))


pos = 0
rows = []
while True:
    i = data.find(b".symbol", pos)
    if i < 0:
        break
    j = data.find(b".kd", i, i + 300)
    if j < 0:
        pos = i + 1
        continue
    name = data[i + len(b".symbol") + 2:j].decode("utf-8", "replace")
    if want in name:
        tag = name.split(want)[1].split("EEEv")[0] if want in name else name

        def back(key):   # the keys before .symbol in the record (alphabetical order); the fixstr prefix is part of the match
            kb = key if isinstance(key, (bytes, bytearray)) else key.encode()
            k = data.rfind(b"\xab" + kb, max(0, i - 800), i)
            if k < 0:
                return None
            j2 = k + 1 + len(kb)
            return data[j2 + 1] if data[j2] == 0xCC else data[j2]

        def fwd(key):    # the keys after it (.symbol .kd .uniform_work_group_size ... .wavefront_size)
            k = data.find(b"\xab" + key if len(key) < 32 else key, j, j + 900)
            if k < 0:
                return None
            v = data[k + 1 + len(key)]
            return data[k + 2 + len(key)] if v == 0xCC else v

        rows.append((tag, fwd(b".vgpr_count"), fwd(b".vgpr_spill_count"), fwd(b".sgpr_count"),
                     back(b".group_segment_fixed_size"), back(b".private_segment_fixed_size")))
    pos = j + 3

for r in sorted(rows):
    print("%-26s vgpr=%-4s spill=%-3s sgpr=%-4s lds=%-6s priv=%-3s" % r)
print("kernels:", len(rows))
