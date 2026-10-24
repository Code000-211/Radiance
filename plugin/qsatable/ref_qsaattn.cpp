/* plugin/qsatable/ref_qsaattn.cpp -- the host row: the definition of `qsa_attn`.
 *
 * The plainest way that is right: walk the row's keys in the order `rad_qsa_attn_keys` states, score
 * them in a first pass, take the maximum, then accumulate in the same order -- libref's ref_attn_paged
 * shape, in double. Nothing here assumes anything about the table beyond the contract: the union
 * ascends in block order and the mask is a filter.
 *
 * Cost is O(rows * heads * keys * head_dim), which is the price of being obvious.
 */
#include "qsaattn.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

int64_t elem_bytes(uint32_t dt) {
    const int bits = rad_dtype_bits(dt);
    return bits % 8 == 0 ? bits / 8 : 0;
}

}  /* namespace */

int rad_ref_qsa_attn(const RadArgs* a) {
    RadQsaAttnCall c{};
    const int rc = rad_qsa_attn_args(a, &c);
    if (rc != RAD_OK) return rc;
    const int64_t esz_q  = elem_bytes(c.q_dtype);
    const int64_t esz_kv = elem_bytes(c.kv_dtype);
    const int64_t esz_o  = elem_bytes(c.out_dtype);
    if (!esz_q || !esz_kv || !esz_o) return RAD_E_UNSUPPORTED;

    std::vector<float>   qf((size_t)c.head_dim);
    std::vector<double>  acc((size_t)c.head_dim);
    std::vector<float>   sc;
    for (int64_t row = 0; row < c.M; ++row) {
        const int64_t s = c.seq[row];
        if (s < 0) return RAD_E_INVAL;
        for (int64_t h = 0; h < c.n_head; ++h) {
            const int64_t kh = h / c.gqa;
            const char* qrow = (const char*)c.q + (row * c.q_pitch + h * c.head_dim) * esz_q;
            char* orow = (char*)c.out + (row * c.out_pitch + h * c.head_dim) * esz_o;
            for (int64_t d = 0; d < c.head_dim; ++d) qf[(size_t)d] = rad_load_f32(qrow, c.q_dtype, d);

            /* Pass one: every visible key's score, in key order, and the maximum. */
            sc.clear();
            double mx = -INFINITY;
            rad_qsa_attn_keys(c, row, [&](int64_t, int64_t page, int64_t first, int64_t last) {
                for (int64_t t = first; t <= last; ++t) {
                    const char* kb = (const char*)c.kv +
                                     (page * c.kv_sb + kh * c.kv_sh + t * c.kv_so) * esz_kv;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < c.head_dim; ++d)
                        dot += qf[(size_t)d] * rad_load_f32(kb, c.kv_dtype, d * c.kv_sd);
                    sc.push_back(dot * c.scale);
                    if (sc.back() > mx) mx = sc.back();
                }
            });
            /* No visible key gives a zero row, not NaN, as every kernel in the project does. */
            if (mx == -INFINITY) {
                for (int64_t d = 0; d < c.head_dim; ++d) rad_store_f32(orow, c.out_dtype, d, 0.0f);
                continue;
            }
            /* Pass two: exponentiate and accumulate in the same order. */
            for (int64_t d = 0; d < c.head_dim; ++d) acc[(size_t)d] = 0.0;
            double sum = 0.0;
            int64_t k = 0;
            rad_qsa_attn_keys(c, row, [&](int64_t, int64_t page, int64_t first, int64_t last) {
                for (int64_t t = first; t <= last; ++t, ++k) {
                    const double w = std::exp((double)sc[(size_t)k] - mx);
                    sum += w;
                    const char* vb = (const char*)c.kv +
                                     (page * c.kv_sb + kh * c.kv_sh + t * c.kv_so +
                                      c.head_dim) * esz_kv;
                    for (int64_t d = 0; d < c.head_dim; ++d)
                        acc[(size_t)d] += w * rad_load_f32(vb, c.kv_dtype, d * c.kv_sd);
                }
            });
            const double inv = sum > 0.0 ? 1.0 / sum : 0.0;
            for (int64_t d = 0; d < c.head_dim; ++d)
                rad_store_f32(orow, c.out_dtype, d, (float)(acc[(size_t)d] * inv));
        }
    }
    return RAD_OK;
}
