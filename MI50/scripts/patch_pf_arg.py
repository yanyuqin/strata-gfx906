#!/usr/bin/env python3
# Give I8Args its prefetch depth (STRATA_I8_PF, default 2 = the 3-stage pipeline).
import io, sys

path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()

old = 'cap_entries / bm + cap_groups, i8_env_int("STRATA_I8_BISECT", 0), i8_n8_takes(i8_switch_parse(std::getenv("STRATA_I8_N8")), tm)};'
new = ('cap_entries / bm + cap_groups, i8_env_int("STRATA_I8_BISECT", 0), i8_n8_takes(i8_switch_parse(std::getenv("STRATA_I8_N8")), tm),\n'
       '             i8_env_int("STRATA_I8_PF", 2) >= 2 ? 2 : 1};   // OWN-I8 prefetch depth (2 = three stages in flight: dot, staged, loading)')

if old not in src:
    sys.exit("anchor not found")
if src.count(old) != 1:
    sys.exit("anchor not unique: %d" % src.count(old))
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched", path)
