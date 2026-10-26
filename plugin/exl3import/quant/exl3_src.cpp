/* exl3_src.cpp -- see exl3_src.h. */
#include "exl3_src.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <sys/stat.h>
#include <unistd.h>

namespace exl3 {

/* ================================================================== JSON */

const Json* Json::get(const char* key) const {
    if (kind != Obj) return nullptr;
    for (const auto& kv : obj)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

namespace {

struct Parser {
    const char* p;
    const char* end;
    std::string* why;

    bool fail(const char* what) {
        if (why) *why = std::string("malformed JSON: ") + what;
        return false;
    }
    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }
    bool lit(const char* s) {
        const size_t n = std::strlen(s);
        if ((size_t)(end - p) < n || std::memcmp(p, s, n) != 0) return false;
        p += n;
        return true;
    }
    static void utf8(uint32_t c, std::string* s) {
        if (c < 0x80) { s->push_back((char)c); return; }
        if (c < 0x800) { s->push_back((char)(0xC0 | (c >> 6))); }
        else if (c < 0x10000) { s->push_back((char)(0xE0 | (c >> 12))); s->push_back((char)(0x80 | ((c >> 6) & 0x3F))); }
        else {
            s->push_back((char)(0xF0 | (c >> 18)));
            s->push_back((char)(0x80 | ((c >> 12) & 0x3F)));
            s->push_back((char)(0x80 | ((c >> 6) & 0x3F)));
        }
        s->push_back((char)(0x80 | (c & 0x3F)));
    }
    bool hex4(uint32_t* v) {
        if (end - p < 4) return false;
        *v = 0;
        for (int i = 0; i < 4; ++i, ++p) {
            const char c = *p;
            *v = *v * 16 + (uint32_t)(c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10
                                      : c >= 'A' && c <= 'F' ? c - 'A' + 10 : 99);
            if (*v > 0xFFFF) return false;
        }
        return true;
    }
    bool string(std::string* s) {
        if (p >= end || *p != '"') return fail("expected a string");
        ++p;
        while (p < end && *p != '"') {
            if (*p != '\\') { s->push_back(*p++); continue; }
            if (++p >= end) break;
            const char e = *p++;
            switch (e) {
            case '"': case '\\': case '/': s->push_back(e); break;
            case 'b': s->push_back('\b'); break;
            case 'f': s->push_back('\f'); break;
            case 'n': s->push_back('\n'); break;
            case 'r': s->push_back('\r'); break;
            case 't': s->push_back('\t'); break;
            case 'u': {
                uint32_t c = 0, lo = 0;
                if (!hex4(&c)) return fail("bad \\u escape");
                if (c >= 0xD800 && c < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                    p += 2;
                    if (!hex4(&lo) || lo < 0xDC00 || lo > 0xDFFF) return fail("bad surrogate pair");
                    c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
                }
                utf8(c, s);
                break;
            }
            default: return fail("bad escape");
            }
        }
        if (p >= end) return fail("unterminated string");
        ++p;
        return true;
    }
    bool value(Json* v, int depth) {
        if (depth > 64) return fail("nested deeper than 64");
        ws();
        if (p >= end) return fail("unexpected end");
        if (*p == '{') {
            ++p;
            v->kind = Json::Obj;
            ws();
            if (p < end && *p == '}') { ++p; return true; }
            for (;;) {
                JsonMember kv;
                ws();
                if (!string(&kv.first)) return false;
                ws();
                if (p >= end || *p++ != ':') return fail("expected ':'");
                if (!value(&kv.second, depth + 1)) return false;
                v->obj.push_back(std::move(kv));
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == '}') { ++p; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (*p == '[') {
            ++p;
            v->kind = Json::Arr;
            ws();
            if (p < end && *p == ']') { ++p; return true; }
            for (;;) {
                v->arr.emplace_back();
                if (!value(&v->arr.back(), depth + 1)) return false;
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == ']') { ++p; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (*p == '"') { v->kind = Json::Str; return string(&v->str); }
        if (lit("true")) { v->kind = Json::Bool; v->b = true; return true; }
        if (lit("false")) { v->kind = Json::Bool; v->b = false; return true; }
        if (lit("null")) { v->kind = Json::Null; return true; }
        {
            char* stop = nullptr;
            std::string num;
            while (p < end && std::strchr("+-0123456789.eE", *p) && *p) num.push_back(*p++);
            if (num.empty()) return fail("unexpected character");
            v->kind = Json::Num;
            v->num = std::strtod(num.c_str(), &stop);
            if (!stop || *stop) return fail("bad number");
            return true;
        }
    }
};

bool as_int(const Json* j, int64_t* out) {
    if (!j || j->kind != Json::Num || j->num != (double)(int64_t)j->num) return false;
    *out = (int64_t)j->num;
    return true;
}

}  // namespace

bool parse_json(const char* p, const char* end, Json* out, std::string* why) {
    Parser ps{p, end, why};
    if (!ps.value(out, 0)) return false;
    ps.ws();
    return ps.p == end || ps.fail("trailing characters");
}

/* ================================================================== files */

bool read_file(const std::string& path, std::string* out, std::string* why) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (why) *why = path + ": " + std::strerror(errno);
        return false;
    }
    char buf[1 << 16];
    size_t n;
    out->clear();
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    const bool ok = !std::ferror(f);
    std::fclose(f);
    if (!ok && why) *why = path + ": read error";
    return ok;
}

static bool exists(const std::string& path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

/* ================================================================== the checkpoint */

Source* Source::open(const std::string& dir, std::string* why) {
    static std::mutex mu;
    static std::map<std::string, std::unique_ptr<Source>> opened;
    const uint16_t one = 1;
    if (*(const uint8_t*)&one != 1) {
        if (why) *why = "this host is big-endian; safetensors are little-endian and are read as stored";
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(mu);
    auto it = opened.find(dir);
    if (it != opened.end()) return it->second.get();

    auto s = std::make_unique<Source>();
    s->dir_ = dir;
    std::string text;
    Json cfg;
    if (!read_file(dir + "/config.json", &text, why) ||
        !parse_json(text.data(), text.data() + text.size(), &cfg, why)) {
        if (why) *why = "src=" + dir + ": config.json: " + *why;
        return nullptr;
    }
    const Json* qc = cfg.get("quantization_config");
    const Json* qm = qc ? qc->get("quant_method") : nullptr;
    if (!qm || qm->kind != Json::Str || qm->str != "exl3") {
        if (why)
            *why = "src=" + dir + " is not an EXL3 checkpoint: its config.json has quantization_config." +
                   "quant_method " + (qm && qm->kind == Json::Str ? "'" + qm->str + "'" : "absent") +
                   ", not 'exl3'";
        return nullptr;
    }
    const Json* tc = cfg.get("text_config");
    if (!as_int((tc ? tc : &cfg)->get("num_hidden_layers"), &s->n_layers_) || s->n_layers_ < 1) {
        if (why) *why = "src=" + dir + ": config.json has no text_config.num_hidden_layers";
        return nullptr;
    }
    const std::string index = dir + "/model.safetensors.index.json";
    if (exists(index)) {
        Json ix;
        if (!read_file(index, &text, why) || !parse_json(text.data(), text.data() + text.size(), &ix, why)) {
            if (why) *why = index + ": " + *why;
            return nullptr;
        }
        const Json* wm = ix.get("weight_map");
        if (!wm || wm->kind != Json::Obj) {
            if (why) *why = index + " has no weight_map";
            return nullptr;
        }
        for (const auto& kv : wm->obj) {
            if (kv.second.kind != Json::Str || kv.second.str.find('/') != std::string::npos) {
                if (why) *why = index + ": weight_map['" + kv.first + "'] is not a file in the directory";
                return nullptr;
            }
            s->where_[kv.first] = dir + "/" + kv.second.str;
        }
    } else if (exists(dir + "/model.safetensors")) {
        std::lock_guard<std::mutex> sl(s->mu_);
        if (!s->load_shard(dir + "/model.safetensors", why)) return nullptr;
        for (const auto& kv : s->shards_[dir + "/model.safetensors"]) s->where_[kv.first] = kv.second.file;
    } else {
        if (why) *why = "src=" + dir + " has neither model.safetensors.index.json nor model.safetensors";
        return nullptr;
    }
    Source* raw = s.get();
    opened[dir] = std::move(s);
    return raw;
}

bool Source::load_shard(const std::string& file, std::string* why) {
    if (shards_.count(file)) return true;
    FILE* f = std::fopen(file.c_str(), "rb");
    uint8_t len8[8];
    if (!f || std::fread(len8, 1, 8, f) != 8) {
        if (f) std::fclose(f);
        if (why) *why = file + ": cannot read a safetensors header";
        return false;
    }
    uint64_t n = 0;
    for (int i = 7; i >= 0; --i) n = n << 8 | len8[i];
    struct stat st {};
    if (::stat(file.c_str(), &st) != 0 || n > (uint64_t)st.st_size - 8 || n > (1ull << 30)) {
        std::fclose(f);
        if (why) *why = file + ": a safetensors header length of " + std::to_string(n) + " bytes does not fit";
        return false;
    }
    std::string head((size_t)n, '\0');
    const bool got = std::fread(&head[0], 1, (size_t)n, f) == (size_t)n;
    std::fclose(f);
    Json h;
    if (!got || !parse_json(head.data(), head.data() + head.size(), &h, why) || h.kind != Json::Obj) {
        if (why) *why = file + ": " + (got ? *why : std::string("short read"));
        return false;
    }
    auto& tensors = shards_[file];
    for (const auto& kv : h.obj) {
        if (kv.first == "__metadata__") continue;
        const Json* dt = kv.second.get("dtype");
        const Json* sh = kv.second.get("shape");
        const Json* off = kv.second.get("data_offsets");
        Tensor t;
        int64_t b = 0, e = 0;
        t.file = file;
        if (!dt || dt->kind != Json::Str || !sh || sh->kind != Json::Arr || !off || off->kind != Json::Arr ||
            off->arr.size() != 2 || !as_int(&off->arr[0], &b) || !as_int(&off->arr[1], &e) || b < 0 || e < b) {
            shards_.erase(file);
            if (why) *why = file + ": header entry '" + kv.first + "' is malformed";
            return false;
        }
        for (const Json& d : sh->arr) {
            int64_t v = 0;
            if (!as_int(&d, &v) || v < 0) {
                shards_.erase(file);
                if (why) *why = file + ": '" + kv.first + "' has a malformed shape";
                return false;
            }
            t.shape.push_back(v);
        }
        t.dtype = dt->str;
        t.begin = 8 + n + (uint64_t)b;
        t.end = 8 + n + (uint64_t)e;
        if (t.end > (uint64_t)st.st_size) {
            shards_.erase(file);
            if (why) *why = file + ": '" + kv.first + "' runs past the end of the file";
            return false;
        }
        tensors[kv.first] = std::move(t);
    }
    return true;
}

int Source::find(const std::string& name, Tensor* out, std::string* why) {
    std::lock_guard<std::mutex> lk(mu_);
    auto w = where_.find(name);
    if (w == where_.end()) return 0;
    if (!load_shard(w->second, why)) return -1;
    auto& tensors = shards_[w->second];
    auto t = tensors.find(name);
    if (t == tensors.end()) {
        if (why) *why = "the index puts '" + name + "' in " + w->second + ", whose header does not have it";
        return -1;
    }
    *out = t->second;
    return 1;
}

std::vector<std::string> Source::names() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> out;
    out.reserve(where_.size());
    for (const auto& kv : where_) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

bool Source::read(const Tensor& t, void* dst, std::string* why) const {
    const int fd = ::open(t.file.c_str(), O_RDONLY | O_CLOEXEC);
    uint64_t done = 0;
    if (fd < 0) {
        if (why) *why = t.file + ": " + std::strerror(errno);
        return false;
    }
    while (done < t.bytes()) {
        const ssize_t r = ::pread(fd, (char*)dst + done, (size_t)(t.bytes() - done), (off_t)(t.begin + done));
        if (r <= 0) break;
        done += (uint64_t)r;
    }
    ::close(fd);
    if (done != t.bytes() && why) *why = t.file + ": short read";
    return done == t.bytes();
}

}  // namespace exl3
