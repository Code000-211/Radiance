/* radfile.cpp -- the .rad reader and writer.
 *
 * Two halves of one decision. Everything the reader assumes on the hot path, the writer proves
 * while writing and the reader re-proves at open. The pair is in one file so a change to the
 * layout cannot land on only one side.
 */
#include "radfile.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

namespace rad {

std::string rad_unit_name(int32_t layer, int32_t expert) {
    if (layer < 0)  return "model";
    if (expert < 0) return fmt("layer %d", layer);
    return fmt("layer %d expert %d", layer, expert);
}

/* ================================================================== reader */

RadFile::~RadFile() { close(); }

void RadFile::close() {
    if (base_) ::munmap(base_, (size_t)size_);
    if (fd_ >= 0) ::close(fd_);
    base_ = nullptr; size_ = 0; fd_ = -1;
    hdr_ = nullptr; meta_ = nullptr; dir_ = nullptr; planes_ = nullptr; encs_ = nullptr;
    prof_ = nullptr;
    vocab_ = VocabSection{};
    groups_.clear();
    by_name_.clear();
    path_.clear();
}

int RadFile::read_at(void* dst, int64_t bytes, int64_t off) const {
    if (fd_ < 0) { RAD_ERR("read_at on a closed container"); return RAD_E_STATE; }
    if (off < 0 || bytes < 0 || off + bytes > size_) {
        RAD_ERR("read_at [%lld, %lld) is outside the %lld-byte container",
                (long long)off, (long long)(off + bytes), (long long)size_);
        return RAD_E_FORMAT;
    }
    uint8_t* p = (uint8_t*)dst;
    while (bytes > 0) {
        const ssize_t got = ::pread(fd_, p, (size_t)bytes, (off_t)off);
        if (got < 0) {
            if (errno == EINTR) continue;
            RAD_ERR("%s: pread at %lld: %s", path_.c_str(), (long long)off, std::strerror(errno));
            return RAD_E_IO;
        }
        if (got == 0) {
            RAD_ERR("%s: pread at %lld returned 0 with %lld bytes still wanted",
                    path_.c_str(), (long long)off, (long long)bytes);
            return RAD_E_IO;
        }
        p += got; off += got; bytes -= got;
    }
    return RAD_OK;
}

void RadFile::drop_blob_cache() const {

    if (!base_ || !hdr_ || fd_ < 0) return;
    const int64_t tables = (int64_t)hdr_->data_off < size_ ? (int64_t)hdr_->data_off : size_;
    if (tables >= size_) return;
    /* The process's mapping first, then the cache behind it: a page still mapped here is one the
     * kernel has to unmap before it can drop it, and doing it in the other order leaves the
     * fadvise with nothing to free. */
    ::madvise(base_ + tables, (size_t)(size_ - tables), MADV_DONTNEED);
    ::posix_fadvise(fd_, (off_t)tables, (off_t)(size_ - tables), POSIX_FADV_DONTNEED);
}

/* Advisory, and deliberately NOT paired with an madvise: the load reads with pread and never
 * maps these pages, so there is nothing of ours to unmap -- and the one region that IS read
 * through the mapping, the n-gram table on Tier::Mapped, is never handed to this. */
void RadFile::drop_range(int64_t off, int64_t bytes) const {

    if (fd_ < 0 || bytes <= 0) return;
    ::posix_fadvise(fd_, (off_t)off, (off_t)bytes, POSIX_FADV_DONTNEED);
}

int RadFile::open(const char* path) {

    close();
    if (!path || !*path) return RAD_E_INVAL;
    path_ = path;

    fd_ = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { RAD_ERR("%s: %s", path, std::strerror(errno)); return RAD_E_IO; }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) { RAD_ERR("%s: %s", path, std::strerror(errno)); close(); return RAD_E_IO; }
    size_ = (int64_t)st.st_size;
    if (size_ < (int64_t)sizeof(RadFileHeader)) {
        RAD_ERR("%s: %lld bytes is shorter than a .rad header (%zu)", path,
                (long long)size_, sizeof(RadFileHeader));
        close();
        return RAD_E_FORMAT;
    }

    void* m = ::mmap(nullptr, (size_t)size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) { RAD_ERR("%s: mmap: %s", path, std::strerror(errno)); close(); return RAD_E_IO; }
    base_ = (uint8_t*)m;

    int s = validate();
    if (s < 0) { close(); return s; }

    /* THE TABLES ARE WANTED NOW; THE BLOB IS READ ONCE, FRONT TO BACK.
     *
     * THE ADVICE IS DELIBERATELY ASYMMETRIC AND MUST STAY THAT WAY.
     *
     * MADV_WILLNEED over the whole file asks the kernel to pull every byte into the page cache.
     * That is right for a container that fits in RAM and ruinous for one that does not: it becomes
     * a readahead storm that evicts the pages it just read and never settles, and the weight load
     * collapses to a small fraction of disk bandwidth with the host thread spinning inside
     * hipMemcpyAsync waiting on a copy engine starved for source pages. The tables are small and
     * read immediately and in full, so they keep WILLNEED; the blob gets NO advice at all.
     *
     * MADV_SEQUENTIAL is the obvious replacement for the blob and is worse than silence: it widens
     * the readahead window and drops pages behind the read pointer, and the load is not one
     * stream. It reads a layer's gate_up experts, then that layer's down experts, then the
     * scales -- three cursors over three regions -- so "behind" for one of them is "just about to
     * be read" for another, and every page is fetched, dropped, and fetched again. Default
     * readahead is what this copy loop is fastest under.
     *
     * The Tier::Mapped weights re-advise their own range MADV_RANDOM once the planner has spoken
     * (engine_bringup.cpp); a later madvise over a sub-range wins for that sub-range, so leaving
     * the blob unadvised here does not take that away. */
    const int64_t tables = (int64_t)hdr_->data_off < size_ ? (int64_t)hdr_->data_off : size_;
    ::madvise(base_, (size_t)tables, MADV_WILLNEED);
    /* A container extended in place (RadWriter::begin_append) keeps its tables PAST the blob;
     * they are wanted now just the same. */
    const int64_t blob_end = (int64_t)(hdr_->data_off + hdr_->data_bytes);
    if ((int64_t)hdr_->str_off >= blob_end && blob_end < size_)
        ::madvise(base_ + (blob_end & ~(int64_t)(RAD_ALIGN_UNIT - 1)),
                  (size_t)(size_ - (blob_end & ~(int64_t)(RAD_ALIGN_UNIT - 1))), MADV_WILLNEED);
    s = index_experts();
    if (s < 0) { close(); return s; }

    by_name_.reserve((size_t)hdr_->dir_count * 2);
    for (int64_t i = 0; i < (int64_t)hdr_->dir_count; ++i)
        by_name_.emplace(std::string_view(str(dir_[i].name)), i);

    return RAD_OK;
}

/* Every offset and count in the header is attacker-shaped input until this function has run.
 * Nothing below this point re-checks, so nothing below this point may be reached without it. */
int RadFile::validate() {
    hdr_ = (const RadFileHeader*)base_;

    if (hdr_->magic != RAD_MAGIC) {
        RAD_ERR("%s: not a .rad container (magic %08x, expected %08x)",
                path_.c_str(), hdr_->magic, RAD_MAGIC);
        return RAD_E_FORMAT;
    }
    if (hdr_->version != RAD_FORMAT_VER) {
        RAD_ERR("%s: .rad version %u, and this build reads only %u, which holds encodings rather "
                "than kernel layouts -- convert it again with rad-convert", path_.c_str(),
                hdr_->version, RAD_FORMAT_VER);
        return RAD_E_FORMAT;
    }
    if ((int64_t)hdr_->file_bytes > size_) {
        RAD_ERR("%s: header claims %llu bytes, the file is %lld -- truncated",
                path_.c_str(), (unsigned long long)hdr_->file_bytes, (long long)size_);
        return RAD_E_FORMAT;
    }

    const uint64_t n = (uint64_t)size_;
    auto span_ok = [&](uint64_t off, uint64_t bytes) {
        return bytes == 0 ? off <= n : (off <= n && bytes <= n - off);
    };
    /* A TABLE IS CHECKED BY ITS COUNT, NEVER BY count * sizeof. Every count in the header is
     * uint64 and attacker-shaped, so the product wraps: meta_count = 0x0AAAAAAAAAAAAAAB times
     * 24 bytes is 8, span_ok passes, and the loop below -- which iterates meta_count, not the
     * byte span -- walks 7.6e17 rows off the end of the mapping. Divide into the space that is
     * left instead, where nothing can wrap. */
    auto span_n = [&](uint64_t off, uint64_t count, uint64_t elem) {
        if (count == 0) return off <= n;
        return off <= n && count <= (n - off) / elem;
    };
    auto need = [&](bool ok, const char* what) {
        if (!ok) RAD_ERR("%s: %s lies outside the %lld-byte file", path_.c_str(), what,
                         (long long)size_);
        return ok;
    };

    if (!need(span_ok(hdr_->str_off, hdr_->str_bytes), "the string blob")) return RAD_E_FORMAT;
    if (hdr_->str_bytes == 0 || base_[hdr_->str_off + hdr_->str_bytes - 1] != 0) {
        RAD_ERR("%s: the string blob is empty or its last byte is not NUL, so a string offset "
                "near its end would read off the end of the mapping", path_.c_str());
        return RAD_E_FORMAT;
    }
    str_bytes_ = hdr_->str_bytes;

    if (!need(span_n(hdr_->meta_off, hdr_->meta_count, sizeof(RadFileKV)), "the metadata table"))
        return RAD_E_FORMAT;
    if (!need(span_n(hdr_->dir_off, hdr_->dir_count, sizeof(RadFileEntry)),
              "the weight directory")) return RAD_E_FORMAT;
    if (!need(span_n(hdr_->plane_off, hdr_->plane_count, sizeof(RadFilePlane)),
              "the plane table")) return RAD_E_FORMAT;
    if (!need(span_n(hdr_->enc_off, hdr_->enc_count, sizeof(RadEncoding)),
              "the encoding table")) return RAD_E_FORMAT;
    if (!need(span_n(hdr_->prof_off, hdr_->prof_count, sizeof(RadFileProfile)),
              "the expert profile")) return RAD_E_FORMAT;
    if (!need(span_ok(hdr_->vocab_off, hdr_->vocab_bytes), "the vocab section")) return RAD_E_FORMAT;
    if (!need(span_ok(hdr_->data_off, hdr_->data_bytes), "the data blob")) return RAD_E_FORMAT;

    if (hdr_->data_bytes && (hdr_->data_off % RAD_ALIGN_POOL) != 0) {
        RAD_ERR("%s: the data blob starts at %llu, which is not %u-aligned -- it could not be "
                "mapped with huge pages", path_.c_str(),
                (unsigned long long)hdr_->data_off, RAD_ALIGN_POOL);
        return RAD_E_FORMAT;
    }

    meta_ = (const RadFileKV*)(base_ + hdr_->meta_off);
    dir_  = (const RadFileEntry*)(base_ + hdr_->dir_off);
    planes_ = (const RadFilePlane*)(base_ + hdr_->plane_off);
    encs_ = (const RadEncoding*)(base_ + hdr_->enc_off);
    prof_ = (const RadFileProfile*)(base_ + hdr_->prof_off);

    auto str_ok = [&](rad_stroff o) { return o < str_bytes_; };

    if (!str_ok(hdr_->arch_id) || !str_ok(hdr_->model_name) || !str_ok(hdr_->quant) ||
        !str_ok(hdr_->recipe) ||
        !str_ok(hdr_->created_by)) {
        RAD_ERR("%s: a header string offset points outside the string blob", path_.c_str());
        return RAD_E_FORMAT;
    }

    for (uint64_t i = 0; i < hdr_->meta_count; ++i) {
        if (!str_ok(meta_[i].key) ||
            (meta_[i].type == RAD_P_STR && !str_ok(meta_[i].v.s))) {
            RAD_ERR("%s: metadata row %llu has a string offset outside the string blob",
                    path_.c_str(), (unsigned long long)i);
            return RAD_E_FORMAT;
        }
    }

    /* An encoding the core cannot read is refused here, by its index, rather than handed to a
     * kernel's layout hook as if it meant something: rad_enc_valid() is the whole contract of the
     * struct -- roles, chains, plane kinds and core dtypes. */
    for (uint64_t i = 0; i < hdr_->enc_count; ++i) {
        RadEncoding e;
        std::memcpy(&e, &encs_[i], sizeof e);
        if (e.scheme[RAD_ENC_STR - 1] || e.transform[RAD_ENC_STR - 1] || !rad_enc_valid(&e)) {
            RAD_ERR("%s: encoding %llu is not one the core can read", path_.c_str(),
                    (unsigned long long)i);
            return RAD_E_FORMAT;
        }
        for (int k = 0; k < e.n_planes; ++k)
            if (e.plane[k].role[RAD_ENC_STR - 1]) {
                RAD_ERR("%s: encoding %llu's plane %d has an unterminated role", path_.c_str(),
                        (unsigned long long)i, k);
                return RAD_E_FORMAT;
            }
    }

    for (uint64_t i = 0; i < hdr_->dir_count; ++i) {
        const RadFileEntry& e = dir_[i];
        if (!str_ok(e.name) || !str_ok(e.quantizer) || !str_ok(e.options)) {
            RAD_ERR("%s: directory row %llu has a string offset outside the string blob",
                    path_.c_str(), (unsigned long long)i);
            return RAD_E_FORMAT;
        }
        if (e.rank == 0 || e.rank > RAD_MAX_RANK) {
            RAD_ERR("%s: '%s' has rank %u; the ABI allows 1..%d", path_.c_str(), str(e.name),
                    e.rank, RAD_MAX_RANK);
            return RAD_E_FORMAT;
        }
        for (uint32_t d = 0; d < e.rank; ++d)
            if (e.shape[d] <= 0) {
                RAD_ERR("%s: '%s' has extent %lld in dim %u", path_.c_str(), str(e.name),
                        (long long)e.shape[d], d);
                return RAD_E_FORMAT;
            }
        if (!span_ok(e.offset, e.bytes)) {
            RAD_ERR("%s: '%s' spans [%llu, %llu) which is outside the %lld-byte file -- truncated",
                    path_.c_str(), str(e.name), (unsigned long long)e.offset,
                    (unsigned long long)(e.offset + e.bytes), (long long)size_);
            return RAD_E_FORMAT;
        }
        if (e.bytes && (e.offset % RAD_ALIGN_UNIT) != 0) {
            RAD_ERR("%s: '%s' starts at %llu, not %u-aligned -- a transfer of it would straddle "
                    "a page boundary needlessly", path_.c_str(), str(e.name),
                    (unsigned long long)e.offset, RAD_ALIGN_UNIT);
            return RAD_E_FORMAT;
        }
        if (e.enc >= hdr_->enc_count) {
            RAD_ERR("%s: '%s' names encoding %u of %llu", path_.c_str(), str(e.name), e.enc,
                    (unsigned long long)hdr_->enc_count);
            return RAD_E_FORMAT;
        }
        const RadEncoding& enc = encs_[e.enc];
        if (e.n_planes != (uint32_t)enc.n_planes || e.plane_first > hdr_->plane_count ||
            e.n_planes > hdr_->plane_count - e.plane_first) {
            RAD_ERR("%s: '%s' names %u planes from %llu of %llu for an encoding of %d",
                    path_.c_str(), str(e.name), e.n_planes, (unsigned long long)e.plane_first,
                    (unsigned long long)hdr_->plane_count, enc.n_planes);
            return RAD_E_FORMAT;
        }
        /* EVERY PLANE IS EXACTLY ITS ENCODING'S SIZE AND INSIDE ITS ENTRY. That is what lets a
         * reader slice a plane by arithmetic and a mover treat the entry as one span. */
        int64_t rows = 0, cols = 0;
        rad_enc_view(e.rank, e.shape, &rows, &cols);
        for (uint32_t k = 0; k < e.n_planes; ++k) {
            const RadFilePlane& p = planes_[e.plane_first + k];
            const int64_t want = rad_enc_plane_bytes(&enc.plane[k], rows, cols);
            if ((int64_t)p.bytes != want || p.offset % RAD_ALIGN_SUB != 0 || p.offset < e.offset ||
                p.offset - e.offset > e.bytes || p.bytes > e.bytes - (p.offset - e.offset)) {
                RAD_ERR("%s: '%s' plane %u ('%s') is %llu bytes at %llu; its encoding over "
                        "[%lld, %lld] makes it %lld, %u-aligned inside [%llu, %llu)",
                        path_.c_str(), str(e.name), k, enc.plane[k].role,
                        (unsigned long long)p.bytes, (unsigned long long)p.offset,
                        (long long)rows, (long long)cols, (long long)want, RAD_ALIGN_SUB,
                        (unsigned long long)e.offset, (unsigned long long)(e.offset + e.bytes));
                return RAD_E_FORMAT;
            }
        }
    }

    /* ---------------------------------------------------------------- vocab */
    if (hdr_->vocab_bytes >= sizeof(RadVocabHeader)) {
        /* THE SECTION IS READ IN PLACE through typed pointers, so the header and every array have
         * to sit on their element's alignment, which is where the writer puts them. A file that
         * does not is damaged, and reading it anyway is a misaligned load for every field. */
        if (hdr_->vocab_off % alignof(RadVocabHeader) != 0) {
            RAD_ERR("%s: the vocab section starts at %llu, which is not %zu-aligned", path_.c_str(),
                    (unsigned long long)hdr_->vocab_off, alignof(RadVocabHeader));
            return RAD_E_FORMAT;
        }
        const RadVocabHeader* v = (const RadVocabHeader*)(base_ + hdr_->vocab_off);
        /* Offset 0 means the array is absent -- an optional one (scores are unigram-only, types
         * default to normal) simply is not written. Nothing real can live at file offset 0, so
         * the sentinel is unambiguous. */
        auto arr_ok = [&](uint64_t off, uint64_t bytes, size_t align) {
            return bytes == 0 || off == 0 ||
                   (off >= hdr_->vocab_off &&
                    off - hdr_->vocab_off <= hdr_->vocab_bytes &&
                    bytes <= hdr_->vocab_bytes - (off - hdr_->vocab_off) &&
                    off % align == 0);
        };
        const uint64_t nt = v->n_tokens, nm = v->n_merges, ns = v->n_steps;
        if (!arr_ok(v->tok_text_off,  nt * sizeof(rad_stroff), alignof(rad_stroff)) ||
            !arr_ok(v->tok_score_off, nt * sizeof(float), alignof(float)) ||
            !arr_ok(v->tok_type_off,  nt, 1) ||
            !arr_ok(v->merge_off,     nm * 2 * sizeof(uint32_t), alignof(uint32_t)) ||
            !arr_ok(v->step_off,      ns * sizeof(RadVocabStep), alignof(RadVocabStep))) {
            RAD_ERR("%s: the vocab section's arrays do not fit inside it, or one is not aligned "
                    "to its element", path_.c_str());
            return RAD_E_FORMAT;
        }
        vocab_.h         = v;
        vocab_.tok_text  = (nt && v->tok_text_off) ? (const rad_stroff*)(base_ + v->tok_text_off) : nullptr;
        vocab_.tok_score = (nt && v->tok_score_off) ? (const float*)(base_ + v->tok_score_off) : nullptr;
        vocab_.tok_type  = (nt && v->tok_type_off)  ? (const uint8_t*)(base_ + v->tok_type_off) : nullptr;
        vocab_.merges    = (nm && v->merge_off) ? (const uint32_t*)(base_ + v->merge_off) : nullptr;
        vocab_.steps     = (ns && v->step_off) ? (const RadVocabStep*)(base_ + v->step_off) : nullptr;

        if (vocab_.tok_text)
            for (uint64_t i = 0; i < nt; ++i)
                if (!str_ok(vocab_.tok_text[i])) {
                    RAD_ERR("%s: vocab token %llu points outside the string blob", path_.c_str(),
                            (unsigned long long)i);
                    return RAD_E_FORMAT;
                }
        if (vocab_.steps)
            for (uint64_t i = 0; i < ns; ++i)
                if (!str_ok(vocab_.steps[i].arg0) || !str_ok(vocab_.steps[i].arg1)) {
                    RAD_ERR("%s: vocab step %llu points outside the string blob", path_.c_str(),
                            (unsigned long long)i);
                    return RAD_E_FORMAT;
                }
        if (vocab_.merges)
            for (uint64_t i = 0; i < nm * 2; ++i)
                if (vocab_.merges[i] >= nt) {
                    RAD_ERR("%s: merge %llu names token %u of %llu", path_.c_str(),
                            (unsigned long long)(i / 2), vocab_.merges[i], (unsigned long long)nt);
                    return RAD_E_FORMAT;
                }
        if (!str_ok(v->chat_template)) {
            RAD_ERR("%s: the chat template offset is outside the string blob", path_.c_str());
            return RAD_E_FORMAT;
        }
    } else if (hdr_->vocab_bytes != 0) {
        RAD_ERR("%s: the vocab section is %llu bytes, shorter than its own header (%zu)",
                path_.c_str(), (unsigned long long)hdr_->vocab_bytes, sizeof(RadVocabHeader));
        return RAD_E_FORMAT;
    }

    return RAD_OK;
}

/* Re-derive the fixed stride and refuse a container that does not have one.
 *
 * This is where the format's central claim is checked. `base + id * stride` is what the mover and
 * the expert kernels do on the critical path, and they do it without a bounds test, so the claim
 * has to be true before anything is handed a pointer. A file whose experts within a layer differ
 * in shape or format is REFUSED here rather than read with a table walk: reading it slowly would
 * quietly return exactly the property the format exists to provide. */
int RadFile::index_experts() {
    const int64_t n = (int64_t)hdr_->dir_count;
    int64_t i = 0;
    std::vector<int32_t> seen_layers;

    while (i < n) {
        if (dir_[i].expert < 0) { ++i; continue; }

        const int32_t layer = dir_[i].layer;
        if (std::find(seen_layers.begin(), seen_layers.end(), layer) != seen_layers.end()) {
            RAD_ERR("%s: layer %d's experts are not contiguous in the directory, so no single "
                    "stride addresses them", path_.c_str(), layer);
            return RAD_E_FORMAT;
        }
        seen_layers.push_back(layer);

        ExpertGroup g;
        g.layer       = layer;
        g.first_index = i;

        /* Expert 0 establishes the shape of every expert in this layer. */
        int64_t j = i;
        while (j < n && dir_[j].layer == layer && dir_[j].expert == 0) ++j;
        g.n_slot = (int32_t)(j - i);
        if (g.n_slot <= 0 || g.n_slot > 8) {
            RAD_ERR("%s: layer %d expert 0 has %d slots; the directory supports 1..8",
                    path_.c_str(), layer, g.n_slot);
            return RAD_E_FORMAT;
        }
        g.base = dir_[i].offset;

        uint64_t unit_end = g.base;
        for (int32_t s = 0; s < g.n_slot; ++s) {
            const RadFileEntry& e = dir_[i + s];
            if (e.slot != s) {
                RAD_ERR("%s: layer %d expert 0's slots are %d at position %d -- the directory must "
                        "list them 0..n-1 in order", path_.c_str(), layer, e.slot, s);
                return RAD_E_FORMAT;
            }
            if (e.offset < g.base) {
                RAD_ERR("%s: layer %d expert 0 slot %d is before the expert's base", path_.c_str(),
                        layer, s);
                return RAD_E_FORMAT;
            }
            g.slot_off[s]   = e.offset - g.base;
            g.slot_bytes[s] = e.bytes;
            unit_end = std::max(unit_end, e.offset + e.bytes);
        }
        g.unit_bytes = unit_end - g.base;

        /* Every further expert must agree, byte for byte, with expert 0's internal layout, and
         * must sit exactly one stride further along. */
        int32_t e_id = 1;
        while (j < n && dir_[j].layer == layer && dir_[j].expert >= 0) {
            if (dir_[j].expert != e_id) {
                RAD_ERR("%s: layer %d's experts are listed out of order (expected %d, found %d)",
                        path_.c_str(), layer, e_id, dir_[j].expert);
                return RAD_E_FORMAT;
            }
            const uint64_t ebase = dir_[j].offset;
            if (e_id == 1) g.stride = ebase - g.base;
            if (ebase != g.base + (uint64_t)e_id * g.stride) {
                RAD_ERR("%s: layer %d expert %d starts at %llu, but base + %d*%llu is %llu -- the "
                        "layer has no fixed stride and cannot be addressed by arithmetic",
                        path_.c_str(), layer, e_id, (unsigned long long)ebase, e_id,
                        (unsigned long long)g.stride,
                        (unsigned long long)(g.base + (uint64_t)e_id * g.stride));
                return RAD_E_FORMAT;
            }
            for (int32_t s = 0; s < g.n_slot; ++s) {
                if (j + s >= n || dir_[j + s].expert != e_id || dir_[j + s].slot != s) {
                    RAD_ERR("%s: layer %d expert %d does not have the same %d slots expert 0 has",
                            path_.c_str(), layer, e_id, g.n_slot);
                    return RAD_E_FORMAT;
                }
                const RadFileEntry& e = dir_[j + s];
                if (e.offset - ebase != g.slot_off[s] || e.bytes != g.slot_bytes[s]) {
                    RAD_ERR("%s: layer %d expert %d slot %d is at +%llu/%llu bytes where expert 0 "
                            "is at +%llu/%llu -- experts in a layer must share a shape and format",
                            path_.c_str(), layer, e_id, s,
                            (unsigned long long)(e.offset - ebase), (unsigned long long)e.bytes,
                            (unsigned long long)g.slot_off[s], (unsigned long long)g.slot_bytes[s]);
                    return RAD_E_FORMAT;
                }
            }
            j += g.n_slot;
            ++e_id;
        }
        g.n_expert = e_id;
        if (g.n_expert == 1) g.stride = align_up((int64_t)g.unit_bytes, RAD_ALIGN_UNIT);

        groups_.push_back(g);
        i = j;
    }
    return RAD_OK;
}

const RadFileEntry* RadFile::find(std::string_view name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &dir_[it->second];
}

const RadFile::ExpertGroup* RadFile::group_for_layer(int32_t layer) const {
    for (const auto& g : groups_) if (g.layer == layer) return &g;
    return nullptr;
}

bool RadFile::get_i(const char* key, int64_t* out) const {
    for (uint64_t i = 0; i < hdr_->meta_count; ++i)
        if (meta_[i].type == RAD_P_INT && std::strcmp(str(meta_[i].key), key) == 0) {
            if (out) *out = meta_[i].v.i;
            return true;
        }
    return false;
}

bool RadFile::get_f(const char* key, double* out) const {
    for (uint64_t i = 0; i < hdr_->meta_count; ++i)
        if (std::strcmp(str(meta_[i].key), key) == 0) {
            if (meta_[i].type == RAD_KV_F64)      { if (out) *out = meta_[i].v.f; return true; }
            if (meta_[i].type == RAD_P_INT)       { if (out) *out = (double)meta_[i].v.i; return true; }
        }
    return false;
}

const char* RadFile::get_s(const char* key, const char* dflt) const {
    for (uint64_t i = 0; i < hdr_->meta_count; ++i)
        if (meta_[i].type == RAD_P_STR && std::strcmp(str(meta_[i].key), key) == 0)
            return str(meta_[i].v.s);
    return dflt;
}

/* ================================================================== writer */

RadWriter::~RadWriter() { if (began_) abort(); }

int RadWriter::begin(const char* path) {
    if (began_) return RAD_E_STATE;
    if (!path || !*path) return RAD_E_INVAL;

    path_       = path;
    spill_path_ = path_ + ".blob.tmp";
    sealed_ = late_ = false;

    fd_out_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd_out_ < 0) { RAD_ERR("%s: %s", path_.c_str(), std::strerror(errno)); return RAD_E_IO; }

    /* The data blob is spilled beside the target and spliced in at finish(). The tables in front
     * of it cannot be sized until every weight has been added, and the blob is the model -- a
     * 1.5 GB checkpoint is not going to be staged in RAM so the header can be back-patched. */
    fd_spill_ = ::open(spill_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    spill_base_ = 0;
    if (fd_spill_ < 0) {
        RAD_ERR("%s: %s", spill_path_.c_str(), std::strerror(errno));
        ::close(fd_out_); fd_out_ = -1;
        return RAD_E_IO;
    }

    began_ = true;
    return RAD_OK;
}

void RadWriter::abort() {
    if (append_) {
        /* The file is the caller's container, not ours to unlink: put its header back -- a no-op
         * unless finish_append() failed inside that last write -- and cut the tail off. */
        if (fd_out_ >= 0) {
            (void)!::pwrite(fd_out_, &old_hdr_, sizeof old_hdr_, 0);
            (void)!::ftruncate(fd_out_, (off_t)old_size_);
            ::close(fd_out_);
        }
        if (fd_spill_ >= 0) ::close(fd_spill_);
        fd_out_ = fd_spill_ = -1;
        began_ = append_ = false;
        return;
    }
    if (fd_out_ >= 0)   ::close(fd_out_);
    if (fd_spill_ >= 0) ::close(fd_spill_);
    if (!path_.empty())       ::unlink(path_.c_str());
    if (!spill_path_.empty()) ::unlink(spill_path_.c_str());
    fd_out_ = fd_spill_ = -1;
    began_ = false;
}

/* ------------------------------------------------------------------ append in place */

/* The processes other than this one that hold the file open or mapped. /proc/<pid>/fd for an open
 * descriptor and /proc/<pid>/maps for a mapping whose descriptor was closed; a process this user
 * cannot inspect is not counted, which is the same reach the caller has over that process. */
static std::vector<long> other_users(dev_t dev, ino_t ino) {
    std::vector<long> who;
    DIR* d = ::opendir("/proc");
    if (!d) return who;
    const long me = (long)::getpid();
    while (dirent* e = ::readdir(d)) {
        char* end = nullptr;
        const long pid = std::strtol(e->d_name, &end, 10);
        if (!end || *end || pid <= 0 || pid == me) continue;
        bool hit = false;
        char dir[64];
        std::snprintf(dir, sizeof dir, "/proc/%ld/fd", pid);
        if (DIR* f = ::opendir(dir)) {
            while (dirent* x = ::readdir(f)) {
                if (x->d_name[0] == '.') continue;
                char link[384];
                std::snprintf(link, sizeof link, "%s/%s", dir, x->d_name);
                struct stat st{};
                if (::stat(link, &st) == 0 && st.st_dev == dev && st.st_ino == ino) {
                    hit = true;
                    break;
                }
            }
            ::closedir(f);
        }
        if (!hit) {
            char maps[64];
            std::snprintf(maps, sizeof maps, "/proc/%ld/maps", pid);
            if (FILE* m = std::fopen(maps, "r")) {
                char line[4096];
                while (!hit && std::fgets(line, sizeof line, m)) {
                    unsigned maj = 0, mnr = 0;
                    unsigned long long inode = 0;
                    if (std::sscanf(line, "%*s %*s %*s %x:%x %llu", &maj, &mnr, &inode) == 3 &&
                        inode == (unsigned long long)ino && maj == major(dev) && mnr == minor(dev))
                        hit = true;
                }
                std::fclose(m);
            }
        }
        if (hit) who.push_back(pid);
    }
    ::closedir(d);
    return who;
}

int RadWriter::begin_append(const char* path, const RadFile& old) {
    if (began_) return RAD_E_STATE;
    if (!path || !*path || !old.is_open()) return RAD_E_INVAL;
    path_ = path;
    spill_path_.clear();

    fd_out_ = ::open(path, O_RDWR | O_CLOEXEC);
    if (fd_out_ < 0) { RAD_ERR("%s: %s", path, std::strerror(errno)); return RAD_E_IO; }
    struct stat st{};
    if (::fstat(fd_out_, &st) != 0) {
        RAD_ERR("%s: %s", path, std::strerror(errno));
        ::close(fd_out_); fd_out_ = -1;
        return RAD_E_IO;
    }
    /* The file `old` describes, unchanged since it was read: its header, byte for byte, and its
     * length. Anything else and the directory being carried over names somebody else's bytes. */
    if (::pread(fd_out_, &old_hdr_, sizeof old_hdr_, 0) != (ssize_t)sizeof old_hdr_ ||
        std::memcmp(&old_hdr_, &old.header(), sizeof old_hdr_) != 0 ||
        (int64_t)st.st_size != old.size()) {
        RAD_ERR("%s: not the container that was read, or it changed since", path);
        ::close(fd_out_); fd_out_ = -1;
        return RAD_E_STATE;
    }
    const std::vector<long> who = other_users(st.st_dev, st.st_ino);
    if (!who.empty()) {
        RAD_ERR("%s is open in %zu other process(es), pid %ld first -- a running engine reads its "
                "header through its mapping, and extending the file rewrites that header. Stop "
                "it first.", path, who.size(), who[0]);
        ::close(fd_out_); fd_out_ = -1;
        return RAD_E_STATE;
    }

    old_size_     = (int64_t)st.st_size;
    old_data_off_ = old_hdr_.data_off;
    append_base_  = (old_size_ + (int64_t)RAD_ALIGN_POOL - 1) / RAD_ALIGN_POOL * RAD_ALIGN_POOL;
    fd_spill_ = ::dup(fd_out_);
    spill_base_ = append_base_;
    if (fd_spill_ < 0) {
        RAD_ERR("%s: %s", path, std::strerror(errno));
        if (fd_spill_ >= 0) ::close(fd_spill_);
        ::close(fd_out_);
        fd_out_ = fd_spill_ = -1;
        return RAD_E_IO;
    }
    append_ = true;
    began_  = true;
    return RAD_OK;
}

int64_t RadWriter::add_existing(const Weight& w, const RadFile& old, const RadFileEntry& e) {
    if (!began_ || !append_) return RAD_E_STATE;
    /* The bytes and the encoding are the old entry's; what the entry SAYS about where it came from
     * and which unit it moves with is the caller's, as a fresh write would record it. */
    RadFileEntry n = e;
    n.name        = intern(w.name);
    n.quantizer   = intern(w.quantizer);
    n.options     = intern(w.options);
    n.enc         = intern_encoding(old.encoding(e));
    n.layer       = w.layer;
    n.expert      = w.expert;
    n.slot        = w.slot;
    n.plane_first = (uint64_t)planes_.size();
    for (uint32_t k = 0; k < e.n_planes; ++k) {
        planes_.push_back(old.plane(e, (int)k));
        written_.push_back((int64_t)old.plane(e, (int)k).bytes);
    }
    dir_.push_back(n);
    dir_abs_.push_back(1);
    return (int64_t)dir_.size() - 1;
}

uint32_t RadWriter::intern_encoding(const RadEncoding& e) {
    /* Zeroed before it is copied in, so the padding and the unused plane slots of a table row
     * are zeros and two converts of one checkpoint compare equal byte for byte -- and compared
     * in that form, so stray bytes in a caller's unused slots do not make a second row. */
    RadEncoding c;
    std::memset(&c, 0, sizeof c);
    std::memcpy(c.scheme, e.scheme, sizeof c.scheme);
    std::memcpy(c.transform, e.transform, sizeof c.transform);
    c.n_planes = e.n_planes;
    for (int k = 0; k < e.n_planes && k < RAD_ENC_MAX_PLANES; ++k) c.plane[k] = e.plane[k];
    for (size_t i = 0; i < encs_.size(); ++i)
        if (rad_enc_equal(&encs_[i], &c)) return (uint32_t)i;
    encs_.push_back(c);
    return (uint32_t)(encs_.size() - 1);
}

rad_stroff RadWriter::intern(std::string_view s) {
    if (s.empty()) return 0;
    std::string key(s);
    auto it = strmap_.find(key);
    if (it != strmap_.end()) return it->second;
    if (sealed_) late_ = true;
    rad_stroff off = (rad_stroff)strblob_.size();
    strblob_.append(key);
    strblob_.push_back('\0');
    strmap_.emplace(std::move(key), off);
    return off;
}

void RadWriter::meta_i(std::string_view k, int64_t v) {
    if (sealed_) late_ = true;
    RadFileKV kv{}; kv.key = intern(k); kv.type = RAD_P_INT; kv.v.i = v; meta_.push_back(kv);
}
void RadWriter::meta_f(std::string_view k, double v) {
    if (sealed_) late_ = true;
    RadFileKV kv{}; kv.key = intern(k); kv.type = RAD_KV_F64; kv.v.f = v; meta_.push_back(kv);
}
void RadWriter::meta_s(std::string_view k, std::string_view v) {
    if (sealed_) late_ = true;
    RadFileKV kv{}; kv.key = intern(k); kv.type = RAD_P_STR; kv.v.s = intern(v); meta_.push_back(kv);
}

void RadWriter::add_profile(int32_t layer, int32_t expert, float share) {
    if (sealed_) late_ = true;
    RadFileProfile p{}; p.layer = layer; p.expert = expert; p.share = share;
    prof_.push_back(p);
}

/* Movement-unit bookkeeping. A unit is one (layer, expert) pair for a routed expert -- its
 * gate/up/down are adjacent because the unit of transfer is an expert, not a tensor -- and one
 * weight for everything else. */
int RadWriter::check_unit_invariant(const Weight& w, uint64_t off, uint64_t bytes) {
    const bool new_unit = (w.layer != cur_layer_ || w.expert != cur_expert_);
    if (new_unit) { cur_layer_ = w.layer; cur_expert_ = w.expert; }

    if (w.expert < 0) { grp_layer_ = -2; return RAD_OK; }

    if (w.layer != grp_layer_) {
        if (w.expert != 0) {
            RAD_ERR("'%s': layer %d's experts start at %d. A layer's experts must be written "
                    "0..n-1 consecutively or the layer has no fixed stride.",
                    w.name.c_str(), w.layer, w.expert);
            return RAD_E_INVAL;
        }
        grp_layer_      = w.layer;
        grp_n_expert_   = 0;
        grp_n_slot_     = 0;
        grp_slot_seen_  = 0;
        grp_first_off_  = off;
        grp_expert_off_ = off;
        grp_stride_     = 0;
        grp_slot_bytes_.clear();
    }

    if (new_unit) {
        if (w.expert != grp_n_expert_) {
            RAD_ERR("'%s': layer %d expert %d follows expert %d. Experts must be written in "
                    "order.", w.name.c_str(), w.layer, w.expert, grp_n_expert_ - 1);
            return RAD_E_INVAL;
        }
        if (grp_n_expert_ == 1) grp_stride_ = off - grp_first_off_;
        if (grp_n_expert_ >= 1 &&
            off != grp_first_off_ + (uint64_t)grp_n_expert_ * grp_stride_) {
            RAD_ERR("'%s': layer %d expert %d landed at %llu, not base + %d*%llu. The experts in "
                    "a layer differ in size, so the layer has no fixed stride.",
                    w.name.c_str(), w.layer, w.expert, (unsigned long long)off,
                    grp_n_expert_, (unsigned long long)grp_stride_);
            return RAD_E_INVAL;
        }
        if (grp_n_expert_ == 1) grp_n_slot_ = grp_slot_seen_;
        if (grp_n_expert_ >= 1 && grp_slot_seen_ != grp_n_slot_) {
            RAD_ERR("'%s': layer %d expert %d has %d slots where expert 0 has %d.",
                    w.name.c_str(), w.layer, grp_n_expert_ - 1, grp_slot_seen_, grp_n_slot_);
            return RAD_E_INVAL;
        }
        grp_expert_off_ = off;
        grp_slot_seen_  = 0;
        ++grp_n_expert_;
    }

    if (w.slot != grp_slot_seen_) {
        RAD_ERR("'%s': layer %d expert %d slot %d follows slot %d. Slots must be written 0..n-1.",
                w.name.c_str(), w.layer, w.expert, w.slot, grp_slot_seen_ - 1);
        return RAD_E_INVAL;
    }

    /* Expert 0 defines the shape of a unit; every later expert must match it slot for slot. The
     * offset arithmetic above cannot catch a size mismatch on its own: an oversized expert still
     * lands at the right base, and only the NEXT one is displaced -- so an oversized LAST expert
     * passes every offset check and leaves a container whose base + id*stride runs off the end of
     * the group. Refusing here means the writer stops while the operator still has the checkpoint
     * open and can fix the manifest. */
    if (grp_n_expert_ == 1) {
        grp_slot_bytes_.push_back(bytes);
    } else if ((size_t)w.slot < grp_slot_bytes_.size() && grp_slot_bytes_[(size_t)w.slot] != bytes) {
        RAD_ERR("'%s': layer %d expert %d slot %d is %llu bytes where expert 0's is %llu. The "
                "experts in a layer differ in size, so the layer has no fixed stride.",
                w.name.c_str(), w.layer, w.expert, w.slot, (unsigned long long)bytes,
                (unsigned long long)grp_slot_bytes_[(size_t)w.slot]);
        return RAD_E_INVAL;
    }

    ++grp_slot_seen_;
    return RAD_OK;
}

int64_t RadWriter::add_weight(const Weight& w) {
    if (!began_ || sealed_) return RAD_E_STATE;
    if (w.name.empty() || w.rank == 0 || w.rank > RAD_MAX_RANK) return RAD_E_INVAL;
    if (!rad_enc_valid(&w.enc)) {
        RAD_ERR("'%s': its encoding is not one the core can read", w.name.c_str());
        return RAD_E_INVAL;
    }
    for (uint32_t d = 0; d < w.rank; ++d)
        if (w.shape[d] <= 0) {
            RAD_ERR("'%s': extent %lld in dim %u", w.name.c_str(), (long long)w.shape[d], d);
            return RAD_E_INVAL;
        }
    if (append_ && w.expert >= 0) {
        RAD_ERR("'%s' is layer %d's expert %d, and a container extended in place cannot take a "
                "new expert: a layer's experts sit at a fixed stride from each other, and one "
                "written past the end joins no layer. Convert to a new file instead.",
                w.name.c_str(), w.layer, w.expert);
        return RAD_E_INVAL;
    }
    /* ONE ENTRY A NAME. A carried-over entry and a new weight of the same name would both be in
     * the directory, and every lookup takes the first -- so the operator's new weight would be
     * silently unreachable while the old one served, which is the one failure this whole route
     * exists to make impossible. */
    for (size_t i = 0; i < dir_.size(); ++i)
        if (std::strcmp(strblob_.c_str() + dir_[i].name, w.name.c_str()) == 0) {
            RAD_ERR("'%s' is already in this container. A weight is stored once; replacing one "
                    "means converting to a new file, not adding a second entry of the same name.",
                    w.name.c_str());
            return RAD_E_DUPLICATE;
        }

    /* Every entry starts on a movement-unit boundary. That costs up to 4 KiB of padding per
     * matrix -- tens of megabytes on a large MoE -- and buys a transfer that never straddles a
     * page needlessly. Its planes follow at RAD_ALIGN_SUB, a cacheline on the architectures in
     * scope: SoA, so a code plane stays dense and a scale plane is its own stream. */
    int64_t rows = 0, cols = 0;
    rad_enc_view(w.rank, w.shape, &rows, &cols);
    const uint64_t off = (uint64_t)align_up(blob_bytes_, RAD_ALIGN_UNIT);
    uint64_t end = off;
    RadFileEntry e{};
    e.plane_first = (uint64_t)planes_.size();
    for (int k = 0; k < w.enc.n_planes; ++k) {
        const int64_t bytes = rad_enc_plane_bytes(&w.enc.plane[k], rows, cols);
        if (bytes <= 0) {
            RAD_ERR("'%s': plane %d ('%s') of its encoding has no size over [%lld, %lld]",
                    w.name.c_str(), k, w.enc.plane[k].role, (long long)rows, (long long)cols);
            planes_.resize((size_t)e.plane_first);
            written_.resize((size_t)e.plane_first);
            return RAD_E_INVAL;
        }
        RadFilePlane p{};
        p.offset = (uint64_t)align_up((int64_t)end, RAD_ALIGN_SUB);
        p.bytes  = (uint64_t)bytes;
        end = p.offset + p.bytes;
        planes_.push_back(p);
        written_.push_back(0);
    }
    const int st = check_unit_invariant(w, off, end - off);
    if (st < 0) {
        planes_.resize((size_t)e.plane_first);
        written_.resize((size_t)e.plane_first);
        return st;
    }

    e.name      = intern(w.name);
    e.quantizer = intern(w.quantizer);
    e.options   = intern(w.options);
    e.enc       = intern_encoding(w.enc);
    e.rank      = w.rank;
    for (uint32_t i = 0; i < RAD_MAX_RANK; ++i) e.shape[i] = i < w.rank ? w.shape[i] : 0;
    e.layer     = w.layer;
    e.expert    = w.expert;
    e.slot      = w.slot;
    e.n_planes  = (uint32_t)w.enc.n_planes;
    e.offset    = off;             /* blob-relative; finish() adds data_off */
    e.bytes     = end - off;
    blob_bytes_ = (int64_t)end;

    dir_.push_back(e);
    dir_abs_.push_back(0);
    return (int64_t)dir_.size() - 1;
}

int64_t RadWriter::add_weight(const Weight& w, const void* const* planes) {
    const int64_t idx = add_weight(w);
    if (idx < 0) return idx;
    for (int k = 0; k < w.enc.n_planes; ++k)
        RAD_TRY(write_plane(idx, k, 0, planes[k], plane_bytes(idx, k)));
    return idx;
}

int64_t RadWriter::plane_bytes(int64_t idx, int k) const {
    if (idx < 0 || idx >= (int64_t)dir_.size() || k < 0 || (uint32_t)k >= dir_[idx].n_planes)
        return RAD_E_INVAL;
    return (int64_t)planes_[dir_[idx].plane_first + (uint64_t)k].bytes;
}

int RadWriter::write_plane(int64_t idx, int k, int64_t off, const void* data, int64_t bytes) {
    if (!began_) return RAD_E_STATE;
    if (idx < 0 || idx >= (int64_t)dir_.size() || dir_abs_[(size_t)idx] || k < 0 ||
        (uint32_t)k >= dir_[idx].n_planes)
        return RAD_E_INVAL;
    const size_t pi = (size_t)(dir_[idx].plane_first + (uint64_t)k);
    const RadFilePlane& p = planes_[pi];
    if (off < 0 || bytes < 0 || (uint64_t)off > p.bytes || (uint64_t)bytes > p.bytes - (uint64_t)off ||
        (bytes > 0 && !data)) {
        RAD_ERR("weight %lld plane %d: %lld bytes at %lld do not fit its %llu", (long long)idx, k,
                (long long)bytes, (long long)off, (unsigned long long)p.bytes);
        return RAD_E_INVAL;
    }
    const uint8_t* q = (const uint8_t*)data;
    int64_t at = spill_base_ + (int64_t)p.offset + off, n = bytes;
    while (n > 0) {
        const ssize_t w = ::pwrite(fd_spill_, q, (size_t)n, (off_t)at);
        if (w < 0) {
            if (errno == EINTR) continue;
            RAD_ERR("%s: %s", append_ ? path_.c_str() : spill_path_.c_str(), std::strerror(errno));
            return RAD_E_IO;
        }
        q += w; n -= w; at += w;
    }
    written_[pi] += bytes;
    return RAD_OK;
}

/* EVERY RESERVED BYTE WAS WRITTEN. A plane left short is a hole of zeros the reader cannot tell
 * from codes -- a quantiser that returned early, a block loop with an off-by-one -- so the file is
 * refused here, naming the weight, rather than written. Counted, not tracked by range: writes that
 * overlapped would pass, and no producer here writes a byte twice. */
int RadWriter::check_complete() const {
    for (size_t i = 0; i < dir_.size(); ++i) {
        if (dir_abs_[i]) continue;
        for (uint32_t k = 0; k < dir_[i].n_planes; ++k) {
            const size_t pi = (size_t)(dir_[i].plane_first + k);
            if (written_[pi] != (int64_t)planes_[pi].bytes) {
                RAD_ERR("'%s' plane %u: %lld of its %llu bytes were written",
                        strblob_.c_str() + dir_[i].name, k, (long long)written_[pi],
                        (unsigned long long)planes_[pi].bytes);
                return RAD_E_STATE;
            }
        }
    }
    return RAD_OK;
}

int RadWriter::set_vocab_section(const void* blob, int64_t bytes) {
    if (!began_ || sealed_) return RAD_E_STATE;
    if (!blob || bytes < (int64_t)sizeof(RadVocabHeader)) {
        RAD_ERR("the vocab section is %lld bytes, shorter than its own header (%zu)",
                (long long)bytes, sizeof(RadVocabHeader));
        return RAD_E_INVAL;
    }
    /* Sanity, here rather than at open: the reader would refuse this file, and a converter that
     * writes a file its own reader refuses has wasted an hour of somebody's time. */
    const RadVocabHeader* h = (const RadVocabHeader*)blob;
    auto fits = [&](uint64_t off, uint64_t n) {
        return n == 0 || (off <= (uint64_t)bytes && n <= (uint64_t)bytes - off);
    };
    if (!fits(h->tok_text_off,  (uint64_t)h->n_tokens * sizeof(rad_stroff)) ||
        !fits(h->tok_score_off, h->tok_score_off ? (uint64_t)h->n_tokens * sizeof(float) : 0) ||
        !fits(h->tok_type_off,  h->tok_type_off ? h->n_tokens : 0) ||
        !fits(h->merge_off,     (uint64_t)h->n_merges * 2 * sizeof(uint32_t)) ||
        !fits(h->step_off,      (uint64_t)h->n_steps * sizeof(RadVocabStep))) {
        RAD_ERR("the vocab section's arrays do not fit inside its %lld bytes. Its offsets must be "
                "SECTION-RELATIVE -- serialise with a section_file_off of 0 and let finish() "
                "rebase them.", (long long)bytes);
        return RAD_E_INVAL;
    }

    vocab_blob_.assign((const uint8_t*)blob, (const uint8_t*)blob + bytes);
    have_vocab_ = true;
    return RAD_OK;
}

/* ------------------------------------------------------------------ finish */
static int write_all(int fd, const void* p, int64_t n) {
    const uint8_t* q = (const uint8_t*)p;
    while (n > 0) {
        ssize_t w = ::write(fd, q, (size_t)n);
        if (w < 0) { if (errno == EINTR) continue; return RAD_E_IO; }
        q += w; n -= w;
    }
    return RAD_OK;
}

static int pwrite_all(int fd, const void* p, int64_t n, int64_t at) {
    const uint8_t* q = (const uint8_t*)p;
    while (n > 0) {
        ssize_t w = ::pwrite(fd, q, (size_t)n, (off_t)at);
        if (w < 0) { if (errno == EINTR) continue; return RAD_E_IO; }
        q += w; n -= w; at += w;
    }
    return RAD_OK;
}

static int pad_file(int fd, int64_t n) {
    static const uint8_t zero[65536] = {0};
    while (n > 0) {
        int64_t c = std::min<int64_t>(n, (int64_t)sizeof zero);
        RAD_TRY(write_all(fd, zero, c));
        n -= c;
    }
    return RAD_OK;
}

/* Sections in header order, 8-aligned, then the blob at a 2 MiB boundary so the mapping can use
 * huge pages. Interns the header's own strings, so it is the last thing that may. */
void RadWriter::layout(RadFileHeader* hp) {
    int64_t off = (int64_t)sizeof(RadFileHeader);
    auto place8 = [&](int64_t bytes) {
        off = align_up(off, 8);
        int64_t at = off;
        off += bytes;
        return at;
    };

    RadFileHeader& h = *hp;
    h = RadFileHeader{};
    h.magic       = RAD_MAGIC;
    h.version     = RAD_FORMAT_VER;
    h.arch_id     = intern(arch_);
    h.model_name  = intern(model_);
    h.quant       = intern(quant_);
    h.recipe      = intern(recipe_);
    h.created_by  = intern(created_by_);

    /* Interning is done: everything after this point only reads the blob. */
    h.str_off    = (uint64_t)place8((int64_t)strblob_.size());
    h.str_bytes  = strblob_.size();
    h.meta_off   = (uint64_t)place8((int64_t)(meta_.size() * sizeof(RadFileKV)));
    h.meta_count = meta_.size();
    h.dir_off    = (uint64_t)place8((int64_t)(dir_.size() * sizeof(RadFileEntry)));
    h.dir_count  = dir_.size();
    h.plane_off  = (uint64_t)place8((int64_t)(planes_.size() * sizeof(RadFilePlane)));
    h.plane_count= planes_.size();
    h.enc_off    = (uint64_t)place8((int64_t)(encs_.size() * sizeof(RadEncoding)));
    h.enc_count  = encs_.size();
    h.vocab_off  = (uint64_t)place8((int64_t)vocab_blob_.size());
    h.vocab_bytes= have_vocab_ ? vocab_blob_.size() : 0;
    h.prof_off   = (uint64_t)place8((int64_t)(prof_.size() * sizeof(RadFileProfile)));
    h.prof_count = prof_.size();

    h.data_off   = (uint64_t)align_up(off, RAD_ALIGN_POOL);
    h.data_bytes = (uint64_t)blob_bytes_;
    h.file_bytes = h.data_off + h.data_bytes;
}

int RadWriter::seal() {
    if (!began_ || append_ || sealed_) return RAD_E_STATE;
    for (int64_t n : written_)
        if (n) {
            RAD_ERR("%s: seal() after a plane was written; the blob is already in the spill",
                    path_.c_str());
            return RAD_E_STATE;
        }
    layout(&hdr_);
    const int fd = ::dup(fd_out_);
    if (fd < 0) { RAD_ERR("%s: %s", path_.c_str(), std::strerror(errno)); return RAD_E_IO; }
    ::close(fd_spill_);
    ::unlink(spill_path_.c_str());
    spill_path_.clear();
    fd_spill_   = fd;
    spill_base_ = (int64_t)hdr_.data_off;
    sealed_     = true;
    return RAD_OK;
}

int RadWriter::finish() {
    if (!began_) return RAD_E_STATE;
    if (late_) {
        RAD_ERR("%s: a table changed after seal() fixed the layout", path_.c_str());
        abort();
        return RAD_E_STATE;
    }
    if (check_complete() < 0) { abort(); return RAD_E_STATE; }
    if (append_) return finish_append();

    RadFileHeader h{};
    if (sealed_) h = hdr_;
    else layout(&h);

    /* Blob-relative to absolute. Both tables were built with offsets into the spill file, which
     * is exactly the blob, so one add each. data_off is 2 MiB-aligned and every entry was already
     * 4 KiB-aligned inside the blob, so the alignments survive the shift. */
    for (auto& e : dir_) e.offset += h.data_off;
    for (auto& p : planes_) p.offset += h.data_off;
    if (have_vocab_) {
        auto* vh = (RadVocabHeader*)vocab_blob_.data();
        if (vh->tok_text_off)  vh->tok_text_off  += h.vocab_off;
        if (vh->tok_score_off) vh->tok_score_off += h.vocab_off;
        if (vh->tok_type_off)  vh->tok_type_off  += h.vocab_off;
        if (vh->merge_off)     vh->merge_off     += h.vocab_off;
        if (vh->step_off)      vh->step_off      += h.vocab_off;
    }

    int st = RAD_OK;
    int64_t at = 0;
    auto emit = [&](const void* p, int64_t bytes, int64_t want_at) {
        if (st < 0) return;
        if (want_at > at) { st = pad_file(fd_out_, want_at - at); if (st < 0) return; at = want_at; }
        st = write_all(fd_out_, p, bytes);
        at += bytes;
    };

    emit(&h, (int64_t)sizeof h, 0);
    emit(strblob_.data(), (int64_t)strblob_.size(), (int64_t)h.str_off);
    emit(meta_.data(), (int64_t)(meta_.size() * sizeof(RadFileKV)), (int64_t)h.meta_off);
    emit(dir_.data(),  (int64_t)(dir_.size()  * sizeof(RadFileEntry)), (int64_t)h.dir_off);
    emit(planes_.data(), (int64_t)(planes_.size() * sizeof(RadFilePlane)), (int64_t)h.plane_off);
    emit(encs_.data(), (int64_t)(encs_.size() * sizeof(RadEncoding)), (int64_t)h.enc_off);
    if (have_vocab_) emit(vocab_blob_.data(), (int64_t)vocab_blob_.size(), (int64_t)h.vocab_off);
    emit(prof_.data(), (int64_t)(prof_.size() * sizeof(RadFileProfile)), (int64_t)h.prof_off);
    if (st < 0) { RAD_ERR("%s: %s", path_.c_str(), std::strerror(errno)); abort(); return st; }

    if (at < (int64_t)h.data_off) {
        st = pad_file(fd_out_, (int64_t)h.data_off - at);
        if (st < 0) { RAD_ERR("%s: %s", path_.c_str(), std::strerror(errno)); abort(); return st; }
        at = (int64_t)h.data_off;
    }

    /* SEALED, THE BLOB IS ALREADY HERE: every plane went to its final offset. The length is set
     * explicitly because the blob's tail may be alignment the planes never wrote. */
    if (sealed_) {
        if (::ftruncate(fd_out_, (off_t)h.file_bytes) != 0 || ::fsync(fd_out_) != 0)
            RAD_ERR("%s: %s", path_.c_str(), std::strerror(errno));
        ::close(fd_out_);   fd_out_ = -1;
        ::close(fd_spill_); fd_spill_ = -1;
        began_ = false;
        return RAD_OK;
    }

    /* Splice the spill in. copy_file_range stays in the kernel and reflinks on a filesystem that
     * can, which on btrfs/xfs makes a 1.5 GB blob nearly free; the read/write fallback is there
     * because tmpfs and older kernels cannot. The spill is exactly the blob's length: every
     * reserved byte was written, and the last of them is the last plane's last byte. */
    if (::lseek(fd_spill_, 0, SEEK_SET) < 0) {
        RAD_ERR("%s: %s", spill_path_.c_str(), std::strerror(errno)); abort(); return RAD_E_IO;
    }
    /* ============================== THE SPILL IS FREED AS IT IS COPIED ==============================
     *
     * A naive splice needs TWICE THE CONTAINER free, because the spill stays whole until the copy
     * finishes and is only then unlinked. At container sizes in the hundreds of gigabytes that is
     * the difference between a convert that fits on the disk and one that runs for half an hour,
     * writes every weight, and then dies on the splice with ENOSPC.
     *
     * Punching a hole in the spill behind the copy gives the blocks back to the filesystem
     * IMMEDIATELY, so the peak is one container plus a grain rather than two containers. ext4,
     * xfs and btrfs all support it; where it is not supported fallocate fails and the copy
     * proceeds exactly as it would without this. It is not an optimisation of the copy -- the
     * bytes still move -- it is a bound on peak disk use. */
    int64_t left = blob_bytes_;
    int64_t done = 0, punched = 0;
    auto punch_behind = [&](bool final_grain) {
#if defined(__linux__) && defined(FALLOC_FL_PUNCH_HOLE)
        /* 256 MiB at a time: enough that the syscall is noise against the copy, small enough that
         * the residual peak overhead is negligible beside the container itself. */
        const int64_t grain = 256ll << 20;
        const int64_t len = done - punched;
        if (len <= 0 || (!final_grain && len < grain)) return;
        if (::fallocate(fd_spill_, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                        (off_t)punched, (off_t)len) == 0) punched = done;
#else
        (void)final_grain;
#endif
    };

#if defined(__linux__) && defined(SYS_copy_file_range)
    while (left > 0) {
        /* AN EXPLICIT INPUT OFFSET, so the punch below knows where the copy has reached and the
         * read() fallback can pick up from the same place. */
        loff_t oi = (loff_t)done;
        ssize_t n = (ssize_t)syscall(SYS_copy_file_range, fd_spill_, &oi,
                                     fd_out_, (loff_t*)nullptr, (size_t)left, 0u);
        if (n <= 0) break;
        left -= n;
        done += n;
        punch_behind(false);
    }
#endif
    if (left > 0) {
        if (::lseek(fd_spill_, (off_t)done, SEEK_SET) < 0) {
            RAD_ERR("%s: %s", spill_path_.c_str(), std::strerror(errno)); abort(); return RAD_E_IO;
        }
        std::vector<uint8_t> buf(1 << 20);
        while (left > 0) {
            ssize_t r = ::read(fd_spill_, buf.data(), (size_t)std::min<int64_t>(left, (int64_t)buf.size()));
            if (r < 0) { if (errno == EINTR) continue; RAD_ERR("%s: %s", spill_path_.c_str(),
                                                               std::strerror(errno)); abort(); return RAD_E_IO; }
            if (r == 0) { RAD_ERR("%s: short read splicing the data blob", spill_path_.c_str());
                          abort(); return RAD_E_IO; }
            st = write_all(fd_out_, buf.data(), r);
            if (st < 0) { RAD_ERR("%s: %s", path_.c_str(), std::strerror(errno)); abort(); return st; }
            left -= r;
            done += r;
            punch_behind(false);
        }
    }
    punch_behind(true);

    if (::fsync(fd_out_) != 0) { RAD_ERR("%s: fsync: %s", path_.c_str(), std::strerror(errno)); }
    ::close(fd_out_);   fd_out_ = -1;
    ::close(fd_spill_); fd_spill_ = -1;
    ::unlink(spill_path_.c_str());
    began_ = false;
    return RAD_OK;
}

/* The new weights are already past the old end (write_plane() put them there); what is left is the
 * tables and, last, the header. */
int RadWriter::finish_append() {
    blob_bytes_ = align_up(blob_bytes_, RAD_ALIGN_UNIT);
    const int64_t blob_end = append_base_ + blob_bytes_;

    int64_t off = blob_end;
    auto place8 = [&](int64_t bytes) {
        off = align_up(off, 8);
        int64_t at = off;
        off += bytes;
        return at;
    };

    RadFileHeader h{};
    h.magic       = RAD_MAGIC;
    h.version     = RAD_FORMAT_VER;
    h.arch_id     = intern(arch_);
    h.model_name  = intern(model_);
    h.quant       = intern(quant_);
    h.recipe      = intern(recipe_);
    h.created_by  = intern(created_by_);

    h.str_off    = (uint64_t)place8((int64_t)strblob_.size());
    h.str_bytes  = strblob_.size();
    h.meta_off   = (uint64_t)place8((int64_t)(meta_.size() * sizeof(RadFileKV)));
    h.meta_count = meta_.size();
    h.dir_off    = (uint64_t)place8((int64_t)(dir_.size() * sizeof(RadFileEntry)));
    h.dir_count  = dir_.size();
    h.plane_off  = (uint64_t)place8((int64_t)(planes_.size() * sizeof(RadFilePlane)));
    h.plane_count= planes_.size();
    h.enc_off    = (uint64_t)place8((int64_t)(encs_.size() * sizeof(RadEncoding)));
    h.enc_count  = encs_.size();
    h.vocab_off  = (uint64_t)place8((int64_t)vocab_blob_.size());
    h.vocab_bytes= have_vocab_ ? vocab_blob_.size() : 0;
    h.prof_off   = (uint64_t)place8((int64_t)(prof_.size() * sizeof(RadFileProfile)));
    h.prof_count = prof_.size();

    /* The blob is the old one and the new weights after it: one span, still pool-aligned at its
     * start, with the old tail's padding inside it. */
    h.data_off   = old_data_off_;
    h.data_bytes = (uint64_t)(blob_end - (int64_t)old_data_off_);
    h.file_bytes = (uint64_t)off;

    for (size_t i = 0; i < dir_.size(); ++i)
        if (!dir_abs_[i]) dir_[i].offset += (uint64_t)append_base_;
    for (size_t i = 0; i < dir_.size(); ++i)
        if (!dir_abs_[i])
            for (uint32_t k = 0; k < dir_[i].n_planes; ++k)
                planes_[dir_[i].plane_first + k].offset += (uint64_t)append_base_;
    if (have_vocab_) {
        auto* vh = (RadVocabHeader*)vocab_blob_.data();
        if (vh->tok_text_off)  vh->tok_text_off  += h.vocab_off;
        if (vh->tok_score_off) vh->tok_score_off += h.vocab_off;
        if (vh->tok_type_off)  vh->tok_type_off  += h.vocab_off;
        if (vh->merge_off)     vh->merge_off     += h.vocab_off;
        if (vh->step_off)      vh->step_off      += h.vocab_off;
    }

    int st = RAD_OK;
    auto put = [&](const void* p, int64_t bytes, uint64_t at) {
        if (st >= 0 && bytes > 0) st = pwrite_all(fd_out_, p, bytes, (int64_t)at);
    };
    put(strblob_.data(), (int64_t)strblob_.size(), h.str_off);
    put(meta_.data(), (int64_t)(meta_.size() * sizeof(RadFileKV)), h.meta_off);
    put(dir_.data(),  (int64_t)(dir_.size()  * sizeof(RadFileEntry)), h.dir_off);
    put(planes_.data(), (int64_t)(planes_.size() * sizeof(RadFilePlane)), h.plane_off);
    put(encs_.data(), (int64_t)(encs_.size() * sizeof(RadEncoding)), h.enc_off);
    if (have_vocab_) put(vocab_blob_.data(), (int64_t)vocab_blob_.size(), h.vocab_off);
    put(prof_.data(), (int64_t)(prof_.size() * sizeof(RadFileProfile)), h.prof_off);
    if (st >= 0 && ::ftruncate(fd_out_, (off_t)off) != 0) st = RAD_E_IO;
    /* Everything the new header names is durable before the header names it. */
    if (st >= 0 && ::fsync(fd_out_) != 0) st = RAD_E_IO;
    if (st < 0) { RAD_ERR("%s: %s", path_.c_str(), std::strerror(errno)); abort(); return st; }

    /* The way back: the old header and the old length. The old tables are still where that
     * header points, so writing it back and truncating restores the file byte for byte. */
    {
        const std::string bak = path_ + ".pre-append";
        const std::string tmp = bak + ".tmp";
        const int fb = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        const int64_t len = old_size_;
        if (fb < 0 || write_all(fb, &old_hdr_, sizeof old_hdr_) < 0 ||
            write_all(fb, &len, sizeof len) < 0 || ::fsync(fb) != 0) {
            RAD_ERR("%s: %s", tmp.c_str(), std::strerror(errno));
            if (fb >= 0) ::close(fb);
            abort();
            return RAD_E_IO;
        }
        ::close(fb);
        if (::rename(tmp.c_str(), bak.c_str()) != 0) {
            RAD_ERR("%s: %s", bak.c_str(), std::strerror(errno));
            abort();
            return RAD_E_IO;
        }
    }

    if (pwrite_all(fd_out_, &h, (int64_t)sizeof h, 0) < 0 || ::fsync(fd_out_) != 0) {
        RAD_ERR("%s: writing the header: %s", path_.c_str(), std::strerror(errno));
        abort();
        return RAD_E_IO;
    }
    ::close(fd_out_);   fd_out_ = -1;
    ::close(fd_spill_); fd_spill_ = -1;
    began_ = append_ = false;
    return RAD_OK;
}

}  /* namespace rad */
