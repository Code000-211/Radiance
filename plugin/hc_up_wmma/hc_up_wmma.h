/* hc_up_wmma.h -- the ported M=3/4 HC-read row's exports, between its device half and its host half.
 *
 * `rad_hc_up_wmma_launch` and the row accessors live in hc_up_wmma.hip (they need the device
 * templates); the schema, the plugin exports and the dispatch-log globals are exports.cpp. Nothing
 * here includes a HIP header, so exports.cpp builds in a card-less tree.
 */
#ifndef RAD_HC_UP_WMMA_H
#define RAD_HC_UP_WMMA_H

#include "rad_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The row: 0 rows unless RADIANCE_HC_UP_WMMA=1, and this library alone on the path never wins the
 * op anyway -- see exports.cpp. Defined in hc_up_wmma.hip, so only a build with a device half has
 * them. */
int                  rad_hc_up_wmma_row_count(void);
const RadKernelInfo* rad_hc_up_wmma_row_at(int i);

#ifdef __cplusplus
}
#endif

#endif  /* RAD_HC_UP_WMMA_H */
