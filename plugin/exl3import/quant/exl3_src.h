/* exl3_src.h -- reading an EXL3 checkpoint directory (turboderp's exllamav3 conversion) one tensor at a time.
 *
 * A directory is config.json (whose quantization_config.quant_method must be "exl3") and safetensors shards,
 * named by model.safetensors.index.json, or one model.safetensors. The index and each shard's header are
 * parsed once and kept (names, dtypes, shapes, offsets: metadata only); a tensor's bytes are read on demand
 * into the caller's buffer and nothing of them is kept. Every call is safe from several threads at once --
 * rad-convert quantises small weights in parallel.
 *
 * Little-endian hosts only: tensor bytes are handed over as stored. Source::open refuses a big-endian host.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace exl3 {

/* ------------------------------------------------------------------ JSON (just what these files need) */
struct JsonMember;
struct Json {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    /* Not std::pair<std::string, Json>: a pair of an incomplete type is not a vector element libstdc++
     * accepts, where a declared struct completed below is (C++17 vector of an incomplete type). */
    std::vector<JsonMember> obj;
    const Json* get(const char* key) const;   /* an object's member, or null */
};
struct JsonMember {
    std::string first;
    Json second;
};
/* Parse [p, end). False with `why` on malformed text or nesting deeper than 64. */
bool parse_json(const char* p, const char* end, Json* out, std::string* why);

/* ------------------------------------------------------------------ the checkpoint */
struct Tensor {
    std::string          file;     /* the shard's path */
    std::string          dtype;    /* safetensors' tag: "I16", "F16", "I32", ... */
    std::vector<int64_t> shape;
    uint64_t             begin = 0, end = 0;   /* absolute byte offsets in `file` */
    uint64_t bytes() const { return end - begin; }
};

class Source {
public:
    /* The directory's Source, opened once per process and kept: false with `why` when it is not an EXL3
     * checkpoint this reader understands. */
    static Source* open(const std::string& dir, std::string* why);

    int64_t n_layers() const { return n_layers_; }   /* text_config.num_hidden_layers */

    /* Every tensor name the checkpoint holds (the index's weight map, or the one shard's header),
     * sorted. Metadata only: nothing is read. */
    std::vector<std::string> names();

    /* Look a tensor up: 1 found, 0 absent, -1 error (`why`). */
    int find(const std::string& name, Tensor* out, std::string* why);
    /* Read its bytes into dst (t.bytes() of them). */
    bool read(const Tensor& t, void* dst, std::string* why) const;

private:
    bool load_shard(const std::string& file, std::string* why);   /* with mu_ held */

    std::string dir_;
    int64_t n_layers_ = -1;
    std::mutex mu_;
    std::unordered_map<std::string, std::string> where_;   /* tensor name -> shard file */
    std::unordered_map<std::string, std::unordered_map<std::string, Tensor>> shards_;
};

/* A whole file's bytes, for config.json and the index. */
bool read_file(const std::string& path, std::string* out, std::string* why);

}  // namespace exl3
