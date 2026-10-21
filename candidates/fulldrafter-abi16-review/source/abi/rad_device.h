/* rad_device.h -- the device layer. Thin on purpose: everything above it is written once, and
 * every backend implements exactly the small set of calls below.
 *
 * The host backend is not a toy. It is what lets the core, the scheduler, the planner, the
 * tokeniser, the server and libref be built and tested on a machine with no accelerator, and it
 * is the same code path RAD_DOMAIN_HOST kernels run on when the planner puts a layer on the CPU
 * (spec §5.1).
 */
#ifndef RAD_DEVICE_H
#define RAD_DEVICE_H

#include "rad_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* RadEvent;

enum { RAD_MEM_DEVICE = 0, RAD_MEM_HOST_PINNED = 1, RAD_MEM_HOST = 2, RAD_MEM_HOST_MAPPED = 3 };

/* WHAT THE ENGINE KNOWS ABOUT A DEVICE, AND IT IS ONLY THIS. Nothing above this layer names a
 * vendor, a product or an instruction set: the scheduler, the planner and the budgets are written
 * against these fields and learn the machine from them at run time. A kernel library is where
 * hardware knowledge belongs, and it declares which devices it serves.
 *
 * Every field carries whatever the backend reports, in the backend's own vocabulary. */
typedef struct RadDeviceProps {
    char     name[128];       /* the product name, for logs and for the tuning cache's key */
    char     arch[32];        /* the backend's name for the architecture, with any feature
                               * suffixes cut -- "gfx1201", "sm_90". This is what a kernel
                               * plugin's build_target is compared against. */
    int      device_id;
    int      n_cu;            /* independent compute units, however the backend counts them */
    int      warp_size;       /* lanes executed in lockstep; 1 on a scalar backend */
    int64_t  vram_bytes;      /* device-local memory, total and unclaimed */
    int64_t  vram_free;
    int64_t  lds_bytes;       /* per-workgroup scratch, for occupancy arithmetic only */
    int      supports_p2p;
    int      is_host_backend; /* 1 when there is no accelerator behind this device */
} RadDeviceProps;

int  rad_dev_count(void);
int  rad_dev_set(int device);
int  rad_dev_props(int device, RadDeviceProps* out);
int  rad_dev_enable_peer(int self, int peer);

void* rad_dev_alloc(int64_t bytes, int kind);
void  rad_dev_free(void* p, int kind);
/* Host-visible pointer for a RAD_MEM_HOST_MAPPED allocation, so the zero-copy execution site can
 * hand the same allocation to both sides (spec §5.1). Null if the kind has no such view. */
void* rad_dev_host_ptr(void* p);
void* rad_dev_device_ptr(void* p);
int   rad_dev_host_register(void* p, int64_t bytes, unsigned flags);
int   rad_dev_host_unregister(void* p);

int  rad_stream_create(RadStream* out, int high_priority);
void rad_stream_destroy(RadStream s);
int  rad_stream_sync(RadStream s);

int  rad_event_create(RadEvent* out);
void rad_event_destroy(RadEvent e);
int  rad_event_record(RadEvent e, RadStream s);
int  rad_event_wait(RadStream s, RadEvent e);      /* stream waits; no host sync */
int  rad_event_query(RadEvent e);                  /* 1 = complete, 0 = pending, <0 = error */
int  rad_event_sync(RadEvent e);
int  rad_event_elapsed_ms(RadEvent a, RadEvent b, float* out);

int  rad_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream s);
int  rad_memcpy_2d_async(void* dst, int64_t dpitch, const void* src, int64_t spitch,
                         int64_t width, int64_t height, RadStream s);
int  rad_memset_async(void* dst, int value, int64_t bytes, RadStream s);

/* The last device error, cleared by reading it. A device fault is not recoverable in-process;
 * the engine exits non-zero rather than serving from a wedged queue (spec §17). */
const char* rad_dev_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* RAD_DEVICE_H */
