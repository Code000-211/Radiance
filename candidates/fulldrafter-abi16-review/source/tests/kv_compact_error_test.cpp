#define rad_memcpy_async test_compact_memcpy_async
#include "arch/common/rad_kv_compact.h"
#undef rad_memcpy_async
#include <cstdio>
#include <cstdlib>
static int calls = 0;
extern "C" int test_compact_memcpy_async(void*, const void*, int64_t, RadStream) {
    ++calls;
    return -77;
}
int main() {
    unsigned char pool[128]{};
    radiance::RadKVSlotMove moves[2]{};
    moves[0].src_slot=3; moves[0].dst_slot=1;
    moves[1].src_slot=4; moves[1].dst_slot=2;
    const int status = radiance::rad_kv_compact_winning_branch(
        pool,128,64,32,4,8,1,2,moves,2,reinterpret_cast<void*>(1));
    if(status != -77 || calls != 1) std::abort();
    std::puts("PASS: first transfer failure propagated; subsequent copies not issued");
}
