/* advance.cpp -- issuing an advance. See advance.h. */
#include "advance_impl.h"

#include "device/device.h"
#include "rad_internal.h"

namespace rad {

int step_advance(const AdvanceArgs& a, RadStream s) {
    if (a.n_seq <= 0) return RAD_OK;
    /* THE HOST BACKEND'S STREAMS ARE QUEUES, each run by its own worker thread (device/host.cpp),
     * and its device memory is host memory. So the rows run here once the stream has finished
     * everything issued before this call, which is where the kernel would run in stream order.
     * Without the wait they can read a sampled token or a staged row whose copy is still queued,
     * and the next step then continues from the token before it. */
    if (device_is_host()) {
        RAD_TRY(rad_stream_sync(s));
        for (int i = 0; i < a.n_seq; ++i) adv_row(a, i);
        return RAD_OK;
    }
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    return rad_step_advance_launch(&a, (void*)s) == 0 ? RAD_OK : RAD_E_DEVICE;
#else
    (void)s;
    return RAD_E_UNSUPPORTED;
#endif
}

}  /* namespace rad */
