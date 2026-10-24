// Reuse the established guarded HC fixture, frozen ABI loader, comparator, event
// timer and negative comparator controls. No upstream test or source is modified.
#define main hc_small_probe_original_main
#include "../hc-small-m-probe/hc_small_m_probe.hip"
#undef main

namespace {
using namespace probe;
constexpr const char* Frozen = "build_hip/radiance_home/kernels/libr4d.so";
constexpr const char* Candidate = "/tmp/hc-down-wmma.so";

void intermediate(const OracleCall& baseline, const Buffer& candidate_scratch, int rows, int band) {
    const Bytes ref = baseline.scratch->get(), got = candidate_scratch.get();
    const size_t dbytes = (size_t)rows * LR * 2, bandbytes = (size_t)band * LR * 2;
    require(ref.size() >= dbytes + rows * kHcMax * 4 &&
            got.size() >= bandbytes + band * kHcMax * 4 + 2 * (N / GROUP) * 4,
            "intermediate scratch too short");
    double num = 0, den = 0, max_abs = 0;
    for (size_t i = 0; i < dbytes / 2; ++i) {
        const float x = host_bf(load<uint16_t>(got, i));
        const float y = host_bf(load<uint16_t>(ref, i));
        require(std::isfinite(x) && std::isfinite(y) && load<uint16_t>(got, i) != 0xcdcd,
                "missing or nonfinite down output");
        const double delta = (double)x - y;
        num += delta * delta; den += (double)y * y;
        max_abs = std::max(max_abs, std::fabs(delta));
    }
    const double relative = den > 0 ? std::sqrt(num / den) : (num == 0 ? 0 : 1);
    std::printf("  d rel_l2=%.9g max_abs=%.9g limit=0.006\n", relative, max_abs);
    require(relative <= 6e-3, "down output exceeds existing BF16 comparator limit");
    for (int r = 0; r < rows; ++r) for (int c = 0; c < HC; ++c) {
        const size_t idx = r * kHcMax + c;
        const float scale = load<float>(got, bandbytes / 4 + idx);
        require(std::isfinite(scale) &&
                load<uint32_t>(got, bandbytes / 4 + idx) == load<uint32_t>(ref, dbytes / 4 + idx),
                "normalized scale differs bitwise from frozen row");
    }
    for (int r = 0; r < rows; ++r)
        for (int c = HC * 4; c < kHcMax * 4; ++c)
            require(got[bandbytes + r * kHcMax * 4 + c] == 0xcd,
                    "normalization scale tail overwritten");
    if (band > rows)
        require(std::all_of(got.begin() + dbytes, got.begin() + bandbytes,
                            [](unsigned char v) { return v == 0xcd; }), "ghost row written");
    for (int i = 0; i < 2 * (N / GROUP); ++i)
        require(load<uint32_t>(got, (bandbytes + band * kHcMax * 4) / 4 + i) == 0,
                "arrival counter not reset after operation");
}

void refuse(OracleCall& call, const Oracle& plugin, hipStream_t stream, int expected) {
    require(plugin.row->launch(&call.args, (RadStream)stream) == expected,
            "candidate accepted invalid geometry, dtype, stride, option, or scratch");
}
void boundaries(OracleCall& call, const Oracle& plugin, hipStream_t stream, int rows) {
    auto p = call.params[0];
    for (int bad : {1, 2, 5, 8}) {
        call.params[0] = pi("M", bad);
        refuse(call, plugin, stream, RAD_E_SHAPE);
    }
    call.params[0] = p;
    if (rows == 4) { call.params[0] = pi("M", 3); refuse(call, plugin, stream, RAD_E_SHAPE); call.params[0] = p; }
    --call.args.scratch_bytes; refuse(call, plugin, stream, RAD_E_SCRATCH); ++call.args.scratch_bytes;
    auto up = call.tensors[3], h = call.tensors[0], x = call.tensors[5], down = call.tensors[2];
    call.tensors[3].dtype = RAD_BF16; refuse(call, plugin, stream, RAD_E_DTYPE); call.tensors[3] = up;
    call.tensors[2].data = (unsigned char*)down.data + 1;
    refuse(call, plugin, stream, RAD_E_STRIDE); call.tensors[2] = down;
    call.tensors[0].shape[0] = 2; refuse(call, plugin, stream, RAD_E_SHAPE); call.tensors[0] = h;
    call.tensors[0].stride[0] += 8; refuse(call, plugin, stream, RAD_E_STRIDE); call.tensors[0] = h;
    --call.tensors[5].shape[0]; refuse(call, plugin, stream, RAD_E_STRIDE); call.tensors[5] = x;
    if (call.tensors[7].data) {
        auto qs = call.tensors[8], rqs = call.tensors[10];
        call.tensors[8].data = nullptr; refuse(call, plugin, stream, RAD_E_INVAL); call.tensors[8] = qs;
        call.tensors[10].data = nullptr; refuse(call, plugin, stream, RAD_E_INVAL); call.tensors[10] = rqs;
    }
}

int run_down_probe(int argc, char** argv) {
    const Options opt = options(argc, argv);
    require(opt.candidate_plugin.empty() && !opt.rowpair && !opt.fused_norm && !opt.up_wmma,
            "use --m/--device/--check-only/--warmup/--iters/--repeats; candidate path is fixed");
    require(!opt.m || opt.m == 3 || opt.m == 4, "only M=3 or M=4");
    HIP(hipSetDevice(opt.device));
#ifdef HC_PROBE_AQL
    require(rad_dev_set(opt.device) == RAD_OK, "native device selection failed");
#endif
    hipDeviceProp_t props{};
    HIP(hipGetDeviceProperties(&props, opt.device));
    require(std::strncmp(props.gcnArchName, "gfx1201", 7) == 0 && props.warpSize == 32 &&
            props.sharedMemPerBlock >= 65536, "requires gfx1201 wave32 and 64 KiB LDS");
    Oracle baseline(Frozen), plugin(Candidate, "hc_down_wmma_m3_m4_probe");
    Stream stream;
    Events events;
    int cases = 0;
    for (int rows : {3, 4}) {
        if (opt.m && rows != opt.m) continue;
        for (uint64_t seed : {uint64_t(1), uint64_t(17), uint64_t(101)}) {
            Inputs input(rows, seed);
            std::array<Snapshot, 3> modes;
            for (int mode = 0; mode < 3; ++mode) {
                Outputs frozen(rows), candidate(rows);
                OracleCall before(baseline, input, frozen, rows, mode);
                OracleCall after(plugin, input, candidate, rows, mode);
                auto ref = [&] { before.launch(baseline, stream.value); };
                auto test = [&] { after.launch(plugin, stream.value); };
                ref(); test(); stream.sync();
                const auto expected = frozen.get(), got = candidate.get();
                std::printf("case M=%d seed=%llu mode=%s\n", rows, (unsigned long long)seed,
                            mode == 0 ? "none" : mode == 1 ? "fp8+rot" : "int8+rot");
                require(compare(expected, got, mode, false, true), "frozen ABI output comparison failed");
                require(expected[1] == got[1], "unchanged inject reduction is not byte exact");
                quant_contract(expected, rows, mode);
                quant_contract(got, rows, mode);
                controls(expected, got, rows, mode, false);
                intermediate(before, *after.scratch, rows, rows);
                input.check();
                boundaries(after, plugin, stream.value, rows);
                ref(); test(); stream.sync();
                require(frozen.get() == expected && candidate.get() == got,
                        "repeated operation changed outputs");
                intermediate(before, *after.scratch, rows, rows);
                RadLaunchDesc desc[2]{};
                int ndesc = 0;
                const auto untouched = after.scratch->get();
                require(plugin.row->describe &&
                        plugin.row->describe(&after.args, (RadStream)stream.value, desc, 1, &ndesc) == RAD_E_FULL &&
                        plugin.row->describe(&after.args, (RadStream)stream.value, desc, 2, &ndesc) == RAD_OK &&
                        ndesc == 2 && after.scratch->get() == untouched,
                        "describe capacity/count or side effects differ");
                for (int i = 0; i < ndesc; ++i)
                    HIP(hipLaunchKernel(desc[i].func,
                        dim3(desc[i].grid[0], desc[i].grid[1], desc[i].grid[2]),
                        dim3(desc[i].block[0], desc[i].block[1], desc[i].block[2]),
                        desc[i].params, desc[i].shared_bytes, stream.value));
                stream.sync();
                require(candidate.get() == got, "described dispatch changed outputs");
                intermediate(before, *after.scratch, rows, rows);
                if (rows == 3 && seed == 1 && mode == 0) {
                    after.params[0] = pi("M", 4);
                    Buffer wide((size_t)plugin.row->scratch(&after.args));
                    after.args.scratch = wide.data(); after.args.scratch_bytes = wide.size;
                    test(); stream.sync();
                    require(candidate.get() == got, "M3 inside band4 changed outputs");
                    intermediate(before, wide, rows, 4);
                    after.params[0] = pi("M", rows);
                    after.args.scratch = after.scratch->data(); after.args.scratch_bytes = after.scratch->size;
                }
                modes[mode] = got;
                if (mode) {
                    require(got[0] == modes[0][0] && got[1] == modes[0][1],
                            "quant mode changed x or inject");
                    if (mode == 2) require(got[4] == modes[1][4] && got[5] == modes[1][5],
                                           "INT8 mode changed rotated FP8 output");
                }
                if (!opt.check_only) {
                    for (int i = 0; i < opt.warmup; ++i) { ref(); test(); }
                    stream.sync();
                    std::vector<double> a, b;
                    for (int repeat = 0; repeat < opt.repeats; ++repeat) {
                        if (repeat & 1) {
                            b.push_back(measure(test, stream.value, events, opt.iters));
                            a.push_back(measure(ref, stream.value, events, opt.iters));
                        } else {
                            a.push_back(measure(ref, stream.value, events, opt.iters));
                            b.push_back(measure(test, stream.value, events, opt.iters));
                        }
                    }
                    require(frozen.get() == expected && candidate.get() == got,
                            "timed operations changed outputs");
                    intermediate(before, *after.scratch, rows, rows);
                    spreads("frozen_libr4d", a);
                    spreads("hc_down_probe", b);
                }
                std::printf("  outputs,guards,counters,description,refusals=PASS\n");
                ++cases;
            }
        }
    }
    std::printf("PASS cases=%d timing=%s (standalone, NOT sealed BetterBench)\n",
                cases, opt.check_only ? "disabled" : "event submissions");
    return 0;
}
} // namespace
int main(int argc, char** argv) {
    try { return run_down_probe(argc, argv); }
    catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
