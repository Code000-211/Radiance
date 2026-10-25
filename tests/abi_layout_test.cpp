/* abi_layout_test.cpp -- the ABI's struct layouts, pinned.
 *
 * Plugins are built apart from the engine, so a struct's layout is the contract: a field that moves
 * or changes size under one RAD_ABI_VERSION is read at the wrong offset by every plugin built
 * before it, and fields are appended without a version bump only because they are trailing and
 * zero-defaulted. The on-disk structs of rad_format.h (and the encoding records serialised into a
 * container) are the .rad format itself.
 *
 * Checked: a C translation unit (abi_c_test.c, strict C11) lays every struct out as C++ does; every
 * struct's listed fields account for its size, so the list in abi_layout_rows.h is complete; and
 * the layout agrees with the table below, taken at RAD_ABI_VERSION 26 and RAD_FORMAT_VER 2. Under
 * those versions a listed field may not move or change size, an ABI struct may only grow by fields
 * past its old end, and an on-disk struct may not change at all. The table is a snapshot of the
 * headers: nothing else states the layout. When it disagrees the test prints the live table.
 */
#include "rad_test.h"

#include "rad_abi.h"
#include "rad_builder.h"
#include "rad_device.h"
#include "rad_encoding.h"
#include "rad_format.h"
#include "rad_philox.h"
#include "rad_planes.h"
#include "rad_plugin.h"
#include "rad_quant.h"
#include "rad_runtime.h"
#include "rad_sample.h"
#include "rad_types.h"
#include "rad_yarn.h"

#include "abi_layout.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

struct Row {
    std::string type, field;
    size_t offset = 0, size = 0, align = 1;
};

/* The C++ view, from the same row list as abi_c_test.c. */
std::vector<Row> cpp_rows() {
    return {
#define ABI_STRUCT(T)   Row{ #T, "", 0, sizeof(T), alignof(T) },
#define ABI_FIELD(T, f) Row{ #T, #f, offsetof(T, f), sizeof(((T*)0)->f), \
                             alignof(decltype(((T*)0)->f)) },
#include "abi_layout_rows.h"
#undef ABI_STRUCT
#undef ABI_FIELD
    };
}

/* The layout at RAD_ABI_VERSION 28 / RAD_FORMAT_VER 2 (see the file header). */
constexpr uint32_t kTableAbi = 28, kTableFormat = 2;
const AbiRow kTable[] = {
#include "abi_layout_table.h"
};

/* The .rad container's records: they may not change at all under one RAD_FORMAT_VER. */
const std::set<std::string> kOnDisk = { "RadEncPlane", "RadEncoding", "RadFileEntry",
                                        "RadFileHeader", "RadFileKV", "RadFilePlane",
                                        "RadFileProfile", "RadVocabHeader", "RadVocabStep" };

void print_table(const std::vector<Row>& rows) {
    fprintf(stderr, "    the live table (abi_layout_table.h):\n");
    for (const Row& r : rows)
        fprintf(stderr, "{ \"%s\", \"%s\", %zu, %zu },\n", r.type.c_str(), r.field.c_str(),
                r.offset, r.size);
}

}  /* namespace */

/* C and C++ lay every struct out alike: the ABI promises plain C, and a plugin written in C reads
 * the same bytes the engine writes. */
TEST(c_and_cpp_lay_out_every_abi_struct_alike) {
    const std::vector<Row> cpp = cpp_rows();
    REQUIRE_EQ(cpp.size(), (size_t)abi_c_n_rows);
    int bad = 0;
    for (size_t i = 0; i < cpp.size(); ++i) {
        const AbiRow& c = abi_c_rows[i];
        if (cpp[i].type != c.type || cpp[i].field != c.field || cpp[i].offset != c.offset ||
            cpp[i].size != c.size) {
            fprintf(stderr, "    %s.%s: C++ %zu+%zu, C %zu+%zu\n", cpp[i].type.c_str(),
                    cpp[i].field.c_str(), cpp[i].offset, cpp[i].size, c.offset, c.size);
            ++bad;
        }
    }
    CHECK_EQ(bad, 0);
    CHECK(cpp.size() > 400);
}

/* Each struct's listed fields tile it: every field starts at the end of the one before, rounded up
 * to its own alignment, and the struct ends at the last field's end rounded up to the struct's. A
 * field missing from abi_layout_rows.h leaves a hole and fails here. */
TEST(the_row_list_accounts_for_every_byte_of_every_struct) {
    std::map<std::string, std::vector<Row>> fields;
    std::map<std::string, Row> whole;
    for (const Row& r : cpp_rows()) {
        if (r.field.empty()) whole[r.type] = r;
        else                 fields[r.type].push_back(r);
    }
    int bad = 0;
    for (const auto& [type, w] : whole) {
        std::vector<Row> f = fields[type];
        std::sort(f.begin(), f.end(),
                  [](const Row& a, const Row& b) { return a.offset < b.offset; });
        size_t end = 0;
        for (const Row& r : f) {
            const size_t want = (end + r.align - 1) / r.align * r.align;
            if (r.offset != want) {
                fprintf(stderr, "    %s.%s at %zu; the fields before it end at %zu\n", type.c_str(),
                        r.field.c_str(), r.offset, end);
                ++bad;
            }
            end = r.offset + r.size;
        }
        if ((end + w.align - 1) / w.align * w.align != w.size) {
            fprintf(stderr, "    %s is %zu bytes; its listed fields end at %zu\n", type.c_str(),
                    w.size, end);
            ++bad;
        }
    }
    CHECK_EQ(bad, 0);
    CHECK(whole.size() >= 45);
}

/* The layout agrees with the table: under the table's versions no listed field moves or changes
 * size, an ABI struct grows only by fields past its old end, and an on-disk struct is unchanged. */
TEST(the_abi_layout_matches_the_table_for_its_version) {
    const std::vector<Row> live = cpp_rows();
    if (RAD_ABI_VERSION != kTableAbi || RAD_FORMAT_VER != kTableFormat) {
        fprintf(stderr, "    the table is for ABI %u / format %u and the headers are ABI %u / "
                        "format %u: regenerate abi_layout_table.h from the printout and update "
                        "kTableAbi / kTableFormat\n", kTableAbi, kTableFormat,
                (unsigned)RAD_ABI_VERSION, (unsigned)RAD_FORMAT_VER);
        print_table(live);
        CHECK(false);
        return;
    }
    std::map<std::pair<std::string, std::string>, const Row*> by;
    for (const Row& r : live) by[{ r.type, r.field }] = &r;
    std::map<std::string, size_t> old_size;
    std::set<std::pair<std::string, std::string>> listed;
    int bad = 0;
    for (const AbiRow& t : kTable) {
        listed.insert({ t.type, t.field });
        if (!*t.field) old_size[t.type] = t.size;
        const auto it = by.find({ t.type, t.field });
        if (it == by.end()) {
            fprintf(stderr, "    %s.%s is gone\n", t.type, t.field);
            ++bad;
            continue;
        }
        const Row& r = *it->second;
        const bool disk = kOnDisk.count(t.type) != 0;
        const bool ok = !*t.field ? (disk ? r.size == t.size : r.size >= t.size)
                                  : r.offset == t.offset && r.size == t.size;
        if (!ok) {
            fprintf(stderr, "    %s%s%s: %zu+%zu, the table has %zu+%zu\n", t.type,
                    *t.field ? "." : "", t.field, r.offset, r.size, t.offset, t.size);
            ++bad;
        }
    }
    /* A field the table does not list is an addition: allowed past the struct's old end, on a
     * struct that is not on disk. */
    for (const Row& r : live) {
        if (r.field.empty() || listed.count({ r.type, r.field })) continue;
        const auto os = old_size.find(r.type);
        if (os == old_size.end()) continue;     /* a struct the table predates */
        if (kOnDisk.count(r.type) || r.offset < os->second) {
            fprintf(stderr, "    %s.%s is new at %zu, inside the struct's old %zu bytes\n",
                    r.type.c_str(), r.field.c_str(), r.offset, os->second);
            ++bad;
        }
    }
    if (bad) print_table(live);
    CHECK_EQ(bad, 0);
}

/* rad_yarn.h's C form computes what its C++ form does, bit for bit, over a sweep of coefficients
 * and rotary indices, and both validate alike: one is what a C plugin's kernel runs, the other
 * what libref, libavx and libr4d run. */
TEST(rad_yarn_is_the_same_function_in_c_and_cpp) {
    int bad = 0, n = 0;
    for (float factor : { 0.0f, 1.0f, 2.0f, 4.0f, 40.0f })
        for (float low : { 0.0f, 3.25f, 10.0f })
            for (float high : { 0.0f, 3.25f, 17.5f, 32.0f }) {
                RadYarn y{};
                y.factor = factor;
                y.low = low;
                y.high = high;
                y.magnitude = 1.07f;
                for (int i = 0; i < 64; ++i)
                    for (float inv : { 1.0f, 0.37f, 1e-4f }) {
                        ++n;
                        const float a = y.frequency(inv, i);
                        const float b = abi_c_yarn_frequency(factor, low, high, 1.07f, inv, i);
                        bad += std::memcmp(&a, &b, sizeof a) != 0;
                    }
            }
    CHECK(n > 10000);
    CHECK_EQ(bad, 0);
    const float cases[][4] = { { 0, 0, 0, 1 }, { 2, 1, 5, 1.1f }, { 0.5f, 1, 5, 1 },
                               { 2, 5, 1, 1 }, { 2, 1, 5, 0 }, { NAN, 1, 5, 1 },
                               { 2, NAN, 5, 1 }, { INFINITY, 1, 5, 1 } };
    for (const auto& c : cases) {
        RadYarn y{};
        y.factor = c[0]; y.low = c[1]; y.high = c[2]; y.magnitude = c[3];
        CHECK_EQ((int)rad_yarn_valid(y), abi_c_yarn_valid(c[0], c[1], c[2], c[3]));
    }
}

RAD_TEST_MAIN()
