/* checkpoint.h -- a safetensors or GGUF checkpoint as a source of logical weights.
 *
 * A checkpoint is read twice in this engine: by rad-convert, which writes a container from it, and
 * by the loader, which serves a checkpoint whose encoding is trivial directly (spec §4.3). Both
 * need the same three things, and this is the one copy of each:
 *
 *   - the tensors, as one row-major namespace however many shards and drafters they came in;
 *   - the model's metadata, from config.json (and its siblings) or the GGUF header;
 *   - a LOGICAL WEIGHT for a declared name: the tensors the architecture plugin's name map
 *     resolves it to -- concatenations, alternatives and expert slices included -- with the
 *     encoding they already are when that encoding is trivial, and the rows of it on demand.
 *
 * Trivial is the set spec §4.3 names: plain f32, f16 and bf16 (GGUF's too), and block-scaled
 * FP8, where `X.weight_scale_inv` is the scale plane of `X.weight`. Anything else is a weight
 * whose served form needs quantisation work, which a recipe does ahead of time.
 */
#pragma once
#include "../rad_internal.h"
#include "../rad_core.h"
#include "safetensors.h"
#include "../text/gguf.h"

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rad {

class Program;

/* ================================================================== tensors */
/* One shape convention from here on: ROW-MAJOR, shape[0] slowest. GGUF stores ne[] fastest-first,
 * so it is reversed on the way in. Getting this backwards transposes every matrix in the model
 * and produces fluent wrong output, so it is done exactly once, in Checkpoint::open. */
struct SrcTensor {
    std::string          name;
    uint32_t             ggml = 0;
    uint32_t             rad_dtype = RAD_DT_INVALID;   /* INVALID: a GGUF block type */
    std::vector<int64_t> shape;                        /* row-major */
    const void*          data = nullptr;
    int64_t              bytes = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return shape.empty() ? 0 : n;
    }
};

/* ================================================================== metadata */
/* The model's metadata as RadModelMeta, with the strings it points at owned here. */
struct MetaOwn {
    std::string arch, name, quant;
    std::vector<std::string> keys, vals;
    std::vector<const char*> kp, vp;
    RadModelMeta m{};

    void kv(const std::string& k, const std::string& v) { keys.push_back(k); vals.push_back(v); }
    /* A key given LAST that must also WIN: a lookup takes the first row with the key, so
     * appending a key the checkpoint already set would write both and read the old one. */
    void set(const std::string& k, const std::string& v);
    bool has(const std::string& k) const;
    const char* get(const std::string& k) const;     /* null when absent */
    /* Republish the pointers RadModelMeta carries. Every kv()/set() may move the strings, so this
     * runs after the last of them and before `m` is read. */
    void finish();
};

int meta_from_config_json(const std::string& dir, MetaOwn* out);
int meta_merge_config_json(const std::string& dir, const std::string& prefix, MetaOwn* out);
int meta_merge_generation_config(const std::string& dir, MetaOwn* out);
int meta_merge_preprocessor_configs(const std::string& dir, MetaOwn* out);
int meta_from_gguf(GgufFile& g, MetaOwn* out);

/* ================================================================== the checkpoint */
class Checkpoint {
public:
    Checkpoint() = default;
    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    /* A .gguf, a .safetensors, a model.safetensors.index.json, or the directory holding one --
     * and the metadata beside it. */
    int open(const std::string& path);
    /* A SECOND checkpoint, merged into this one's namespace and metadata under `prefix`: a
     * drafter that ships as its own repository (DFlash2) borrows the target's embedding and head,
     * so it belongs in the target's container, its tensors under `draft.` and its config.json
     * under `draft.` keys. */
    int add_draft(const std::string& dir, const std::string& prefix);

    const SrcTensor* find(const std::string& n) const;
    const std::vector<SrcTensor>& tensors() const { return tensors_; }
    bool               is_gguf() const { return kind_ == Gguf; }
    GgufFile&          gguf() { return gguf_; }
    const std::string& path() const { return path_; }
    const std::string& dir() const { return dir_; }   /* where config.json and tokenizer.json sit */
    MetaOwn&           meta() { return meta_; }
    const MetaOwn&     meta() const { return meta_; }

    /* Sub-tensor `index` of a STACKED tensor -- dim 0 of a rank >= 2 tensor -- as a view, owned
     * here so every pointer handed out stays valid for the checkpoint's life. Null, with the
     * reason logged, when there is no such slice. */
    const SrcTensor* slice(const SrcTensor& st, int index, const char* declared);

    /* FP8's block, from quantization_config.weight_block_size; 128x128 when it says nothing. */
    int64_t fp8_block_rows() const { return fp8_br_; }
    int64_t fp8_block_cols() const { return fp8_bc_; }

private:
    int add_shard(const std::string& p, const std::string& prefix = "");
    int open_safetensors(const std::string& path);
    int open_gguf(const std::string& path);

    enum Kind { None, Gguf, SafeT } kind_ = None;
    std::string path_, dir_;
    GgufFile    gguf_;
    MetaOwn     meta_;
    std::vector<std::unique_ptr<SafeTensorsFile>> shards_;
    std::vector<SrcTensor> tensors_;
    std::unordered_map<std::string, size_t> index_;
    std::deque<SrcTensor> sliced_;            /* views; a deque never moves what it holds */
    int64_t fp8_br_ = 128, fp8_bc_ = 128;
};

/* ================================================================== logical weights */
/* A declared weight's source, as the name map resolves it: pieces joined along `concat_dim`, and
 * for block-scaled FP8 each piece's scale plane beside it. */
struct CkptWeight {
    std::string name;                         /* the logical (declared) name */
    std::vector<const SrcTensor*> pieces;     /* the codes, in concatenation order */
    std::vector<const SrcTensor*> scales;     /* FP8 only: one a piece */
    std::vector<const SrcTensor*> zeroes;     /* affine Q4 only: one a piece */
    int      concat_dim = 0;
    uint32_t rank = 0;
    int64_t  shape[RAD_MAX_RANK] = {0};       /* the logical shape */
    int64_t  rows = 0, cols = 0;              /* its [rows, cols] view (rad_enc_view) */
    /* The encoding the pieces ARE, when it is trivial; `why` says what is not when it is not. */
    bool        trivial = false;
    RadEncoding enc{};
    std::string why;
};

/* Resolve `name` through `prog`'s name map. RAD_OK; RAD_E_NOTFOUND when the name map has no entry
 * for it yet (declare may add one later) or, for an optional weight, none of its tensors is
 * there; anything else with the reason in `why`. */
int ckpt_resolve(Checkpoint& ck, const Program& prog, const std::string& name, bool optional,
                 CkptWeight* out, std::string* why);

/* The checkpoint tensor a piece or scale reads: its name without the "[index]" a slice of a
 * stacked tensor carries, so an expert reads the stack it was cut from. */
std::string ckpt_tensor_of(const SrcTensor& t);

/* Whether `w` reads any tensor named in `names` (checkpoint names, as ckpt_tensor_of gives them),
 * through a piece or a scale. Each name it reads is added to `hit` when that is given. */
bool ckpt_reads_any(const CkptWeight& w, const std::unordered_set<std::string>& names,
                    std::unordered_set<std::string>* hit = nullptr);

/* Logical rows [r0, r0 + nr) of `w` as f32 [nr, cols], whatever the pieces are stored as: a
 * plain float widened, FP8 times its block scale, a GGUF block type dequantised. */
int ckpt_rows_f32(const CkptWeight& w, int64_t r0, int64_t nr, float* out, std::string* why);

/* Rows [r0, r0 + nr) of plane `k` of a TRIVIAL `w`, in the plane's own elements, dense into
 * `dst`: the bytes a container entry holds for it. */
int ckpt_plane_rows(const CkptWeight& w, int k, int64_t r0, int64_t nr, void* dst,
                    std::string* why);

}  /* namespace rad */
