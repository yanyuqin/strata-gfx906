#!/usr/bin/env python3
# i8_run: set the packed-image launch arguments when the caller passed an image
import io, sys

path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()

old = 'i8_env_int("STRATA_I8_PF", 2) >= 2 ? 2 : 1, (const unsigned char*) pk};'
new = ('i8_env_int("STRATA_I8_PF", 2) >= 2 ? 2 : 1, nullptr, 0, 0, 0};\n'
       '    if (pk != nullptr) {   // strata-pack: the images of this launch\'s experts, in grp_ptr\'s order\n'
       '        a.pk_base = (const unsigned char*) pk;\n'
       '        i8_pk_args(L.gu_type, L.d_type, L, a.pk_stride, a.pk_up_off, a.pk_dn_off);\n'
       '    }')

if src.count(old) != 1:
    sys.exit("anchor count %d" % src.count(old))
io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
print("patched")
