#!/usr/bin/env python3
# Split native_expert_i8 into an internal i8_run(.., pk) and two public entries (blob / packed image), and declare them.
import io, sys

cu = sys.argv[1]
hpp = sys.argv[2]


def sub1(text, old, new, what):
    if text.count(old) != 1:
        sys.exit("%s: anchor count %d" % (what, text.count(old)))
    return text.replace(old, new)


src = io.open(cu, encoding="utf-8").read()

src = sub1(src,
           'void native_expert_i8(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,\n'
           '                      const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,\n'
           '                      void* scratch, float* out, void* stream) {',
           'static void i8_run(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,\n'
           '                   const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,\n'
           '                   void* scratch, float* out, void* stream, const void* pk) {',
           "head")

src = sub1(src,
           'i8_env_int("STRATA_I8_PF", 2) >= 2 ? 2 : 1};',
           'i8_env_int("STRATA_I8_PF", 2) >= 2 ? 2 : 1, (const unsigned char*) pk};',
           "args")

src = sub1(src,
           '    else i8_launch<42, false>(tm, a, s);\n    check("native_expert_i8/down");\n}',
           '    else i8_launch<42, false>(tm, a, s);\n    check("native_expert_i8/down");\n'
           '}\n'
           '\n'
           'void native_expert_i8(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,\n'
           '                      const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,\n'
           '                      void* scratch, float* out, void* stream) {\n'
           '    i8_run(L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, cap_entries, x_q8_1, scratch, out, stream, nullptr);\n'
           '}\n'
           'void native_expert_i8_pk(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,\n'
           '                         const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,\n'
           '                         void* scratch, float* out, void* stream, const void* pk) {\n'
           '    i8_run(L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, cap_entries, x_q8_1, scratch, out, stream, pk);\n'
           '}\n'
           'bool native_expert_i8_pk_supported(const NativeExpertLayout& L) { return i8_pk_supported(L.gu_type, L.d_type); }\n'
           'size_t native_expert_i8_pk_bytes(const NativeExpertLayout& L) { return i8_pk_expert_bytes(L.gu_type, L.d_type, L); }\n'
           'void native_expert_i8_pk_build(const NativeExpertLayout& L, const unsigned long long* grp_ptr, int ne, void* pk, void* stream) {\n'
           '    i8_pk_build(L, grp_ptr, ne, pk, stream);\n'
           '    check("native_expert_i8_pk_build");\n'
           '}',
           "tail")

decl = ('/// strata-pack (the i8 kernels\' packed expert image: [tile][stage][row] records, see i8_expert.cuh): is this layout covered, and how many\n'
        '/// bytes does one expert\'s packed image take.\n'
        'bool native_expert_i8_pk_supported(const NativeExpertLayout& L) ;\n'
        'size_t native_expert_i8_pk_bytes(const NativeExpertLayout& L);\n'
        '/// Builds `ne` experts\' packed images (blobs from grp_ptr, one image of native_expert_i8_pk_bytes each, back to back) on `stream`.\n'
        'void native_expert_i8_pk_build(const NativeExpertLayout& L, const unsigned long long* grp_ptr, int ne, void* pk, void* stream);\n'
        '/// native_expert_i8 reading the packed images instead of the blobs: same numbers, one launch\'s arguments more.\n'
        'void native_expert_i8_pk(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,\n'
        '                         const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,\n'
        '                         void* scratch, float* out, void* stream, const void* pk);\n')

htxt = io.open(hpp, encoding="utf-8").read()
anchor = 'void native_expert_i8(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,\n                      const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,\n                      void* scratch, float* out, void* stream);\n'
htxt = sub1(htxt, anchor, anchor + decl, "hpp")

io.open(cu, "w", encoding="utf-8").write(src)
io.open(hpp, "w", encoding="utf-8").write(htxt)
print("patched")
