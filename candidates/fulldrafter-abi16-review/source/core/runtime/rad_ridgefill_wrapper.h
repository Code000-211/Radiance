/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* rad_ridgefill_wrapper.h -- Radiance-RidgeFill Projector interop and runtime wrapper.
 *
 * Provides complete C++ runtime interfaces for:
 * 1. Mode and flag normalization (--ridgefill, RADIANCE_RIDGEFILL=1/speed/quality).
 * 2. Dynamic binding and validation of all ridgefill.so kernel plugin symbols.
 * 3. Complete Safetensors projector weights inspection, mapping, and verification.
 * 4. Projector bundle management (ridgefill.json manifest, int8 codes/scales/biases,
 *    recurrent GDN state corrections, and token classification tables).
 */
#pragma once

#include "../rad_core.h"
#include <rad_abi.h>
#include <rad_types.h>

#include <nlohmann/json.hpp>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace ridgefill {

/* ---------------------------------------------------------------- Mode & Config */

enum class Mode {
    Off = 0,
    Plumb = 1,
    Speed = 2,
    Quality = 3
};

inline const char* mode_to_string(Mode m) {
    switch (m) {
        case Mode::Off:     return "off";
        case Mode::Plumb:   return "plumb";
        case Mode::Speed:   return "speed";
        case Mode::Quality: return "quality";
    }
    return "off";
}

inline Mode parse_mode(const char* s) {
    if (!s || !*s) return Mode::Off;
    if (!strcasecmp(s, "off") || !strcmp(s, "0") || !strcasecmp(s, "false")) return Mode::Off;
    if (!strcasecmp(s, "speed") || !strcmp(s, "1") || !strcasecmp(s, "on") || !strcasecmp(s, "true")) return Mode::Speed;
    if (!strcasecmp(s, "quality")) return Mode::Quality;
    if (!strcasecmp(s, "plumb")) return Mode::Plumb;
    return Mode::Off;
}

inline uint32_t safetensors_dtype_to_rad(const std::string& s) {
    if (s == "BF16")     return RAD_BF16;
    if (s == "F32")      return RAD_F32;
    if (s == "F16")      return RAD_F16;
    if (s == "I8")       return RAD_I8;
    if (s == "U8")       return RAD_U8;
    if (s == "I16")      return RAD_I16;
    if (s == "I32")      return RAD_I32;
    if (s == "I64")      return RAD_I64;
    if (s == "BOOL")     return RAD_BOOL;
    if (s == "F8_E4M3")  return RAD_F8E4M3;
    if (s == "F8_E5M2")  return RAD_F8E5M2;
    return RAD_DT_INVALID;
}

inline const char* rad_dtype_to_safetensors(uint32_t dt) {
    switch (dt) {
        case RAD_BF16:   return "BF16";
        case RAD_F32:    return "F32";
        case RAD_F16:    return "F16";
        case RAD_I8:     return "I8";
        case RAD_U8:     return "U8";
        case RAD_I16:    return "I16";
        case RAD_I32:    return "I32";
        case RAD_I64:    return "I64";
        case RAD_BOOL:   return "BOOL";
        case RAD_F8E4M3: return "F8_E4M3";
        case RAD_F8E5M2: return "F8_E5M2";
        default:         return "UNKNOWN";
    }
}

struct InteropConfig {
    Mode        mode = Mode::Off;
    int64_t     tail = 2048;            /* exact tail tokens T */
    double      alpha = 1.0;            /* GDN correction strength */
    double      share = 0.25;           /* share of class matches kept exact in quality mode */
    int64_t     min_bulk_rows = 1024;   /* minimum bulk rows before taking approximate path */
    int64_t     ckpt_floor = 0;         /* prefix cache checkpoint exact floor rows */
    std::string projector_dir;          /* path to projector weights directory */
    std::string kernels_so;             /* path to ridgefill.so */

    static InteropConfig from_env_and_flags(const std::string& flag_mode = "",
                                            const std::string& flag_proj = "") {
        InteropConfig c;
        const char* e_mode = std::getenv("RADIANCE_RIDGEFILL");
        if (!flag_mode.empty()) {
            c.mode = parse_mode(flag_mode.c_str());
        } else if (e_mode && *e_mode) {
            c.mode = parse_mode(e_mode);
        }

        const char* e_tail = std::getenv("RADIANCE_RIDGEFILL_TAIL");
        if (e_tail && *e_tail) {
            c.tail = std::atoll(e_tail);
        }

        const char* e_alpha = std::getenv("RADIANCE_RIDGEFILL_ALPHA");
        if (e_alpha && *e_alpha) {
            c.alpha = std::atof(e_alpha);
        }

        const char* e_share = std::getenv("RADIANCE_RIDGEFILL_SHARE");
        if (e_share && *e_share) {
            c.share = std::atof(e_share);
        }

        const char* e_floor = std::getenv("RADIANCE_RIDGEFILL_CKPT_FLOOR");
        if (e_floor && *e_floor) {
            c.ckpt_floor = std::atoll(e_floor);
        }

        if (!flag_proj.empty()) {
            c.projector_dir = flag_proj;
        } else {
            const char* e_proj = std::getenv("RADIANCE_RIDGEFILL_PROJECTOR");
            if (e_proj && *e_proj) c.projector_dir = e_proj;
        }

        return c;
    }

    void apply_to_env() const {
        setenv("RADIANCE_RIDGEFILL", mode_to_string(mode), 1);
        if (!projector_dir.empty()) {
            setenv("RADIANCE_RIDGEFILL_PROJECTOR", projector_dir.c_str(), 1);
        }
        if (tail > 0) {
            setenv("RADIANCE_RIDGEFILL_TAIL", std::to_string(tail).c_str(), 1);
        }
        if (alpha != 1.0) {
            setenv("RADIANCE_RIDGEFILL_ALPHA", std::to_string(alpha).c_str(), 1);
        }
        if (share != 0.25) {
            setenv("RADIANCE_RIDGEFILL_SHARE", std::to_string(share).c_str(), 1);
        }
        if (ckpt_floor > 0) {
            setenv("RADIANCE_RIDGEFILL_CKPT_FLOOR", std::to_string(ckpt_floor).c_str(), 1);
        }
    }

    std::string find_kernel_library() const {
        if (!kernels_so.empty() && access(kernels_so.c_str(), R_OK) == 0) return kernels_so;
        const char* rad_home = std::getenv("RADIANCE_HOME");
        if (rad_home && *rad_home) {
            std::string cand = std::string(rad_home) + "/kernels/ridgefill.so";
            if (access(cand.c_str(), R_OK) == 0) return cand;
        }
        const char* paths[] = {
            "/tmp/rf_kernels_build/kernels/ridgefill.so",
            "/tmp/rf_home/kernels/ridgefill.so",
            "/tmp/rf_kernels_build/radiance_home/kernels/ridgefill.so",
            "/usr/local/share/radiance/kernels/ridgefill.so"
        };
        for (const char* p : paths) {
            if (access(p, R_OK) == 0) return p;
        }
        return "";
    }
};

/* ---------------------------------------------------------------- symbols in ridgefill.so */

struct KernelSymbols {
    void* handle = nullptr;

    /* Plugin lifecycle */
    uint32_t (*abi_version)(void) = nullptr;
    const RadPluginInfo* (*plugin_info)(void) = nullptr;
    int (*kernel_count)(void) = nullptr;
    const RadKernelInfo* (*kernel_at)(int) = nullptr;
    int (*schema_count)(void) = nullptr;
    const RadOpSchema* (*schema_at)(int) = nullptr;
    int (*kernel_concurrent)(int, int) = nullptr;

    /* Forwarding entrypoints (schemas 8 and 9) */
    int (*forward_count)(void) = nullptr;
    const RadKernelInfo* (*forward_at)(int) = nullptr;
    int (*forward_concurrent)(int, int) = nullptr;

    /* Direct dispatch functions & op parse functions */
    int (*mask_device)(const RadArgs* a, RadStream s) = nullptr;
    int (*mask_host)(const RadArgs* a, RadStream s) = nullptr;
    int (*mask_parse)(const RadParam* p, int n_p, const RadOperand* o, int n_o, void* out) = nullptr;

    int (*select_device)(const RadArgs* a, RadStream s) = nullptr;
    int (*select_host)(const RadArgs* a, RadStream s) = nullptr;
    int (*select_parse)(const RadParam* p, int n_p, const RadOperand* o, int n_o, void* out) = nullptr;

    int (*drop_device)(const RadArgs* a, RadStream s) = nullptr;
    int (*drop_host)(const RadArgs* a, RadStream s) = nullptr;
    int (*drop_parse)(const RadParam* p, int n_p, const RadOperand* o, int n_o, void* out) = nullptr;

    int (*rho_device)(const RadArgs* a, RadStream s) = nullptr;
    int (*rho_host)(const RadArgs* a, RadStream s) = nullptr;
    int (*rho_parse)(const RadParam* p, int n_p, const RadOperand* o, int n_o, void* out) = nullptr;

    int (*correct_device)(const RadArgs* a, RadStream s) = nullptr;
    int (*correct_host)(const RadArgs* a, RadStream s) = nullptr;
    int (*correct_parse)(const RadParam* p, int n_p, const RadOperand* o, int n_o, void* out) = nullptr;

    int (*state_read_device)(const RadArgs* a, RadStream s) = nullptr;
    int (*state_read_host)(const RadArgs* a, RadStream s) = nullptr;
    int (*state_read_parse)(const RadParam* p, int n_p, const RadOperand* o, int n_o, void* out) = nullptr;

    int (*hazard_device)(const RadArgs* a, RadStream s) = nullptr;
    int (*hazard_host)(const RadArgs* a, RadStream s) = nullptr;
    int (*hazard_parse)(const RadParam* p, int n_p, const RadOperand* o, int n_o, void* out) = nullptr;

    KernelSymbols() = default;
    ~KernelSymbols() { unload(); }

    KernelSymbols(const KernelSymbols&) = delete;
    KernelSymbols& operator=(const KernelSymbols&) = delete;

    KernelSymbols(KernelSymbols&& o) noexcept { move_from(std::move(o)); }
    KernelSymbols& operator=(KernelSymbols&& o) noexcept {
        if (this != &o) {
            unload();
            move_from(std::move(o));
        }
        return *this;
    }

    void unload() {
        if (handle) {
            dlclose(handle);
            handle = nullptr;
        }
        abi_version = nullptr;
        plugin_info = nullptr;
        kernel_count = nullptr;
        kernel_at = nullptr;
        schema_count = nullptr;
        schema_at = nullptr;
        kernel_concurrent = nullptr;
        forward_count = nullptr;
        forward_at = nullptr;
        forward_concurrent = nullptr;
        mask_device = nullptr; mask_host = nullptr; mask_parse = nullptr;
        select_device = nullptr; select_host = nullptr; select_parse = nullptr;
        drop_device = nullptr; drop_host = nullptr; drop_parse = nullptr;
        rho_device = nullptr; rho_host = nullptr; rho_parse = nullptr;
        correct_device = nullptr; correct_host = nullptr; correct_parse = nullptr;
        state_read_device = nullptr; state_read_host = nullptr; state_read_parse = nullptr;
        hazard_device = nullptr; hazard_host = nullptr; hazard_parse = nullptr;
    }

    bool bind(const std::string& so_path, std::string* err = nullptr) {
        unload();
        handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            if (err) *err = dlerror();
            return false;
        }

        auto load = [this](const char* sym) -> void* {
            return dlsym(handle, sym);
        };

        abi_version        = reinterpret_cast<decltype(abi_version)>(load("rad_plugin_abi_version"));
        plugin_info        = reinterpret_cast<decltype(plugin_info)>(load("rad_plugin_info"));
        kernel_count       = reinterpret_cast<decltype(kernel_count)>(load("rad_kernel_count"));
        kernel_at          = reinterpret_cast<decltype(kernel_at)>(load("rad_kernel_at"));
        schema_count       = reinterpret_cast<decltype(schema_count)>(load("rad_kernel_schema_count"));
        schema_at          = reinterpret_cast<decltype(schema_at)>(load("rad_kernel_schema_at"));
        kernel_concurrent  = reinterpret_cast<decltype(kernel_concurrent)>(load("rad_kernel_concurrent"));

        forward_count      = reinterpret_cast<decltype(forward_count)>(load("ridgefill_forward_count"));
        forward_at         = reinterpret_cast<decltype(forward_at)>(load("ridgefill_forward_at"));
        forward_concurrent = reinterpret_cast<decltype(forward_concurrent)>(load("ridgefill_forward_concurrent"));

        mask_device        = reinterpret_cast<decltype(mask_device)>(load("ridgefill_mask_device"));
        mask_host          = reinterpret_cast<decltype(mask_host)>(load("ridgefill_mask_host"));
        mask_parse         = reinterpret_cast<decltype(mask_parse)>(load("ridgefill_mask_parse"));

        select_device      = reinterpret_cast<decltype(select_device)>(load("ridgefill_select_device"));
        select_host        = reinterpret_cast<decltype(select_host)>(load("ridgefill_select_host"));
        select_parse       = reinterpret_cast<decltype(select_parse)>(load("ridgefill_select_parse"));

        drop_device        = reinterpret_cast<decltype(drop_device)>(load("ridgefill_drop_device"));
        drop_host          = reinterpret_cast<decltype(drop_host)>(load("ridgefill_drop_host"));
        drop_parse         = reinterpret_cast<decltype(drop_parse)>(load("ridgefill_drop_parse"));

        rho_device         = reinterpret_cast<decltype(rho_device)>(load("ridgefill_rho_device"));
        rho_host           = reinterpret_cast<decltype(rho_host)>(load("ridgefill_rho_host"));
        rho_parse          = reinterpret_cast<decltype(rho_parse)>(load("ridgefill_rho_parse"));

        correct_device     = reinterpret_cast<decltype(correct_device)>(load("ridgefill_correct_device"));
        correct_host       = reinterpret_cast<decltype(correct_host)>(load("ridgefill_correct_host"));
        correct_parse      = reinterpret_cast<decltype(correct_parse)>(load("ridgefill_correct_parse"));

        state_read_device  = reinterpret_cast<decltype(state_read_device)>(load("ridgefill_state_read_device"));
        state_read_host    = reinterpret_cast<decltype(state_read_host)>(load("ridgefill_state_read_host"));
        state_read_parse   = reinterpret_cast<decltype(state_read_parse)>(load("ridgefill_state_read_parse"));

        hazard_device      = reinterpret_cast<decltype(hazard_device)>(load("ridgefill_hazard_device"));
        hazard_host        = reinterpret_cast<decltype(hazard_host)>(load("ridgefill_hazard_host"));
        hazard_parse       = reinterpret_cast<decltype(hazard_parse)>(load("ridgefill_hazard_parse"));

        if (!abi_version || abi_version() != RAD_ABI_VERSION) {
            if (err) *err = "ridgefill.so ABI version mismatch (expected " + std::to_string(RAD_ABI_VERSION) + ")";
            unload();
            return false;
        }

        if (!kernel_count || kernel_count() < 14) {
            if (err) *err = "ridgefill.so kernel count unexpected (expected at least 14)";
            unload();
            return false;
        }

        if (!schema_count || schema_count() != 9) {
            if (err) *err = "ridgefill.so schema count unexpected (expected 9)";
            unload();
            return false;
        }

        return true;
    }

private:
    void move_from(KernelSymbols&& o) noexcept {
        handle = o.handle;
        abi_version = o.abi_version;
        plugin_info = o.plugin_info;
        kernel_count = o.kernel_count;
        kernel_at = o.kernel_at;
        schema_count = o.schema_count;
        schema_at = o.schema_at;
        kernel_concurrent = o.kernel_concurrent;
        forward_count = o.forward_count;
        forward_at = o.forward_at;
        forward_concurrent = o.forward_concurrent;
        mask_device = o.mask_device; mask_host = o.mask_host; mask_parse = o.mask_parse;
        select_device = o.select_device; select_host = o.select_host; select_parse = o.select_parse;
        drop_device = o.drop_device; drop_host = o.drop_host; drop_parse = o.drop_parse;
        rho_device = o.rho_device; rho_host = o.rho_host; rho_parse = o.rho_parse;
        correct_device = o.correct_device; correct_host = o.correct_host; correct_parse = o.correct_parse;
        state_read_device = o.state_read_device; state_read_host = o.state_read_host; state_read_parse = o.state_read_parse;
        hazard_device = o.hazard_device; hazard_host = o.hazard_host; hazard_parse = o.hazard_parse;
        o.handle = nullptr;
    }
};

/* ---------------------------------------------------------------- safetensors tensor mapping */

struct MappedTensor {
    std::string          name;
    uint32_t             dtype = RAD_DT_INVALID;
    std::string          dtype_name;
    std::vector<int64_t> shape;
    int64_t              numel = 0;
    int64_t              bytes = 0;
    const uint8_t*       data = nullptr;

    template <typename T>
    const T* as() const {
        return reinterpret_cast<const T*>(data);
    }
};

class SafetensorsFile {
public:
    SafetensorsFile() = default;
    ~SafetensorsFile() { close(); }

    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    SafetensorsFile(SafetensorsFile&& o) noexcept { move_from(std::move(o)); }
    SafetensorsFile& operator=(SafetensorsFile&& o) noexcept {
        if (this != &o) {
            close();
            move_from(std::move(o));
        }
        return *this;
    }

    void close() {
        if (data_ && size_ > 0) {
            munmap(const_cast<void*>(reinterpret_cast<const void*>(data_)), size_);
            data_ = nullptr;
            size_ = 0;
        }
        tensors_.clear();
        metadata_.clear();
    }

    bool open_and_map(const std::string& path, std::string* err = nullptr) {
        close();
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            if (err) *err = "cannot open file: " + path;
            return false;
        }

        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size < 8) {
            ::close(fd);
            if (err) *err = "file too short or stat failed: " + path;
            return false;
        }

        size_ = static_cast<size_t>(st.st_size);
        void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) {
            data_ = nullptr;
            size_ = 0;
            if (err) *err = "mmap failed for: " + path;
            return false;
        }

        data_ = static_cast<const uint8_t*>(p);
        if (!parse_header(err)) {
            close();
            return false;
        }
        return true;
    }

    const MappedTensor* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

    const std::map<std::string, MappedTensor>& tensors() const { return tensors_; }
    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    size_t file_size() const { return size_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    std::map<std::string, MappedTensor> tensors_;
    std::map<std::string, std::string> metadata_;

    void move_from(SafetensorsFile&& o) noexcept {
        data_ = o.data_;
        size_ = o.size_;
        tensors_ = std::move(o.tensors_);
        metadata_ = std::move(o.metadata_);
        o.data_ = nullptr;
        o.size_ = 0;
    }

    bool parse_header(std::string* err) {
        if (size_ < 8) {
            if (err) *err = "file size smaller than header length field";
            return false;
        }
        uint64_t header_len = 0;
        std::memcpy(&header_len, data_, 8);
        if (header_len > size_ - 8) {
            if (err) *err = "header length (" + std::to_string(header_len) +
                            ") exceeds available file size (" + std::to_string(size_ - 8) + ")";
            return false;
        }

        std::string json_str(reinterpret_cast<const char*>(data_ + 8), static_cast<size_t>(header_len));
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(json_str);
        } catch (const std::exception& e) {
            if (err) *err = std::string("failed to parse safetensors JSON header: ") + e.what();
            return false;
        }

        if (!j.is_object()) {
            if (err) *err = "safetensors JSON header root is not an object";
            return false;
        }

        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::string& tname = it.key();
            const auto& val = it.value();

            if (tname == "__metadata__") {
                if (val.is_object()) {
                    for (auto m_it = val.begin(); m_it != val.end(); ++m_it) {
                        if (m_it.value().is_string()) {
                            metadata_[m_it.key()] = m_it.value().get<std::string>();
                        }
                    }
                }
                continue;
            }

            if (!val.is_object()) {
                if (err) *err = "tensor entry '" + tname + "' is not an object";
                return false;
            }

            MappedTensor tensor;
            tensor.name = tname;

            if (!val.contains("dtype") || !val["dtype"].is_string()) {
                if (err) *err = "tensor '" + tname + "' missing valid 'dtype'";
                return false;
            }
            tensor.dtype_name = val["dtype"].get<std::string>();
            tensor.dtype = safetensors_dtype_to_rad(tensor.dtype_name);
            if (tensor.dtype == RAD_DT_INVALID) {
                if (err) *err = "tensor '" + tname + "' has unsupported dtype: " + tensor.dtype_name;
                return false;
            }

            if (!val.contains("shape") || !val["shape"].is_array()) {
                if (err) *err = "tensor '" + tname + "' missing valid 'shape'";
                return false;
            }
            tensor.numel = 1;
            for (const auto& dim : val["shape"]) {
                if (!dim.is_number_integer()) {
                    if (err) *err = "tensor '" + tname + "' has non-integer shape dimension";
                    return false;
                }
                int64_t d = dim.get<int64_t>();
                if (d < 0) {
                    if (err) *err = "tensor '" + tname + "' has negative shape dimension";
                    return false;
                }
                tensor.shape.push_back(d);
                tensor.numel *= d;
            }

            if (!val.contains("data_offsets") || !val["data_offsets"].is_array() || val["data_offsets"].size() != 2) {
                if (err) *err = "tensor '" + tname + "' missing valid 2-element 'data_offsets'";
                return false;
            }

            uint64_t off_start = val["data_offsets"][0].get<uint64_t>();
            uint64_t off_end   = val["data_offsets"][1].get<uint64_t>();

            if (off_start > off_end) {
                if (err) *err = "tensor '" + tname + "' has start offset > end offset";
                return false;
            }

            tensor.bytes = static_cast<int64_t>(off_end - off_start);
            int64_t expected_bytes = rad_dtype_bytes(tensor.dtype, tensor.numel);
            if (tensor.bytes != expected_bytes) {
                if (err) *err = "tensor '" + tname + "' byte size (" + std::to_string(tensor.bytes) +
                                ") does not match shape elements * dtype (" + std::to_string(expected_bytes) + ")";
                return false;
            }

            uint64_t tensor_file_start = 8 + header_len + off_start;
            uint64_t tensor_file_end   = 8 + header_len + off_end;
            if (tensor_file_end > size_) {
                if (err) *err = "tensor '" + tname + "' data offset exceeds file boundaries";
                return false;
            }

            tensor.data = data_ + tensor_file_start;
            tensors_[tname] = tensor;
        }

        return true;
    }
};

/* ---------------------------------------------------------------- Projector Bundle */

class ProjectorBundle {
public:
    ProjectorBundle() = default;

    bool open(const std::string& dir, std::string* err = nullptr) {
        dir_ = dir;
        std::string manifest_path = dir + "/ridgefill.json";
        std::ifstream in(manifest_path, std::ios::binary);
        if (!in.is_open()) {
            if (err) *err = "cannot open manifest: " + manifest_path;
            return false;
        }

        nlohmann::json manifest;
        try {
            in >> manifest;
        } catch (const std::exception& e) {
            if (err) *err = std::string("failed to parse ridgefill.json: ") + e.what();
            return false;
        }

        if (!manifest.contains("format") || manifest["format"].get<int>() != 1) {
            if (err) *err = "manifest is not format 1";
            return false;
        }

        name_ = manifest.value("name", "");
        adapter_ = manifest.value("adapter", "");
        if (manifest.contains("projector") && manifest["projector"].is_object()) {
            const auto& p = manifest["projector"];
            dtype_ = p.value("dtype", "i8");
            split_ = p.value("split", 24);
        } else {
            dtype_ = "i8";
            split_ = 24;
        }

        /* Map correction.safetensors if present */
        std::string corr_path = dir + "/correction.safetensors";
        if (access(corr_path.c_str(), R_OK) == 0) {
            if (!correction_file_.open_and_map(corr_path, err)) {
                return false;
            }
        }

        /* Map rowsel.safetensors if present */
        std::string rowsel_path = dir + "/rowsel.safetensors";
        if (access(rowsel_path.c_str(), R_OK) == 0) {
            if (!rowsel_file_.open_and_map(rowsel_path, err)) {
                return false;
            }
        }

        /* Map per-layer files */
        for (int64_t l = split_; l < 48; ++l) {
            std::string fname;
            if (dtype_ == "i8") {
                fname = dir + "/proj8.L" + std::to_string(l) + ".safetensors";
            } else {
                fname = dir + "/proj.L" + std::to_string(l) + ".safetensors";
            }
            if (access(fname.c_str(), R_OK) == 0) {
                auto sf = std::make_unique<SafetensorsFile>();
                if (!sf->open_and_map(fname, err)) {
                    return false;
                }
                layers_[l] = std::move(sf);
            }
        }

        return true;
    }

    const std::string& name() const { return name_; }
    const std::string& dtype() const { return dtype_; }
    bool is_int8() const { return dtype_ == "i8"; }
    int64_t split() const { return split_; }
    const std::string& directory() const { return dir_; }

    bool get_layer_i8(int64_t layer, const int8_t** codes, const uint16_t** scale, const uint16_t** bias,
                      int64_t* n = nullptr, int64_t* k = nullptr) const {
        auto it = layers_.find(layer);
        if (it == layers_.end() || !it->second) return false;
        const auto* tc = it->second->find("proj." + std::to_string(layer) + ".codes");
        const auto* ts = it->second->find("proj." + std::to_string(layer) + ".scale");
        const auto* tb = it->second->find("proj." + std::to_string(layer) + ".bias");
        if (!tc || !ts || !tb) return false;
        if (codes) *codes = tc->as<int8_t>();
        if (scale) *scale = ts->as<uint16_t>();
        if (bias)  *bias  = tb->as<uint16_t>();
        if (n && tc->shape.size() >= 2) *n = tc->shape[0];
        if (k && tc->shape.size() >= 2) *k = tc->shape[1];
        return true;
    }

    bool get_layer_bf16(int64_t layer, const uint16_t** weight, const uint16_t** bias,
                        int64_t* n = nullptr, int64_t* k = nullptr) const {
        auto it = layers_.find(layer);
        if (it == layers_.end() || !it->second) return false;
        const auto* tw = it->second->find("proj." + std::to_string(layer) + ".weight");
        const auto* tb = it->second->find("proj." + std::to_string(layer) + ".bias");
        if (!tw || !tb) return false;
        if (weight) *weight = tw->as<uint16_t>();
        if (bias)   *bias   = tb->as<uint16_t>();
        if (n && tw->shape.size() >= 2) *n = tw->shape[0];
        if (k && tw->shape.size() >= 2) *k = tw->shape[1];
        return true;
    }

    const MappedTensor* get_correction(int64_t layer) const {
        return correction_file_.find("st." + std::to_string(layer));
    }

    const MappedTensor* get_rowsel(const std::string& name = "score") const {
        return rowsel_file_.find(name);
    }

private:
    std::string dir_;
    std::string name_;
    std::string adapter_;
    std::string dtype_ = "i8";
    int64_t split_ = 24;
    SafetensorsFile correction_file_;
    SafetensorsFile rowsel_file_;
    std::map<int64_t, std::unique_ptr<SafetensorsFile>> layers_;
};

}  /* namespace ridgefill */
}  /* namespace rad */
