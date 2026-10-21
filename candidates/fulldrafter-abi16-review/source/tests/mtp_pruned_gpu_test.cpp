// Link mtp_pruned_tree_test.cpp with RAD_MTP_PRUNED_GPU_TEST for the shared suite.
#include "core/device/device.h"
#include "core/plugin/registry.h"

#include <cstdio>
#include <stdexcept>
#include <string>

int rad_run_mtp_pruned_behavior(rad::Registry&, RadStream, bool);

namespace {
void require(bool condition, const std::string& why) {
    if (!condition) throw std::runtime_error(why);
}
void check(int status, const std::string& why) {
    require(status == RAD_OK, why + ": " + rad_strerror(status));
}
struct Session {
    RadStream stream = nullptr;
    Session() {
        check(rad_dev_set(0), "select GPU 0");
        RadDeviceProps properties{};
        check(rad_dev_props(0, &properties), "GPU properties");
        require(!properties.is_host_backend, "a real GPU is required; host backend execution is not a pass");
        check(rad_stream_create(&stream, 0), "create test stream");
    }
    ~Session() { if (stream) rad_stream_destroy(stream); }
};
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 9 && std::string(argv[1]) == "--mine" && std::string(argv[3]) == "--ref" &&
                    std::string(argv[5]) == "--libr4d" && std::string(argv[7]) == "--libref",
                "usage: mtp_pruned_gpu_test --mine PATH --ref PATH --libr4d PATH --libref PATH");
        Session session;
        rad::Registry registry;
        registry.set_code_device(0);
        check(registry.load_plugin(argv[2], 0), "load mine");
        check(registry.load_plugin(argv[4], 1), "load mine_ref");
        check(registry.load_plugin(argv[6], 2), "load libr4d");
        check(registry.load_plugin(argv[8], 3), "load libref");
        check(registry.check_complete(), "complete plugin schemas");
        return rad_run_mtp_pruned_behavior(registry, session.stream, true);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
