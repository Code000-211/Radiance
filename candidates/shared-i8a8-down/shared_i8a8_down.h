/* shared_i8a8_down.h -- the little the candidate's two translation units share.
 *
 * The ABI's export set is exactly rad_plugin_abi_version, rad_plugin_info and the four
 * kernel/schema accessors (rad_abi.h). They carry SI8_EXPORT. Everything else in the DSO --
 * this row's launch, its operand description, and the int8 weight-layout hooks borrowed from
 * libr4d -- is internal, and says so: a hidden definition cannot interpose on libr4d.so's copy of
 * the same hook when both are loaded.
 */
#ifndef SHARED_I8A8_DOWN_H
#define SHARED_I8A8_DOWN_H

#include "rad_abi.h"

#if defined(__GNUC__) || defined(__clang__)
#define SI8_EXPORT __attribute__((visibility("default")))
#define SI8_LOCAL  __attribute__((visibility("hidden")))
#else
#define SI8_EXPORT
#define SI8_LOCAL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The launch and its operand description, defined in shared_i8a8_down.hip. */
SI8_LOCAL int rad_shared_i8a8_down(const RadArgs* a, RadStream stream);
SI8_LOCAL int rad_shared_i8a8_down_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

/* libr4d's int8 block layout, compiled into this DSO from libr4d/r4d_fp8_layout.cpp. Defined
 * there, hidden there, reached only through the row's layout/relayout/unrelayout pointers. */
SI8_LOCAL int r4d_rad_layout_i8_block(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel,
                                      const RadTensor* planes, int n_planes, RadLayout* out);
SI8_LOCAL int r4d_rad_relayout_i8_block(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* planes, int n_planes, void* dst,
                                        int64_t dst_bytes);
SI8_LOCAL int r4d_rad_unrelayout_i8_block(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const void* src, int64_t src_bytes,
                                          const RadTensor* planes, int n_planes);

#ifdef __cplusplus
}
#endif

#endif /* SHARED_I8A8_DOWN_H */
