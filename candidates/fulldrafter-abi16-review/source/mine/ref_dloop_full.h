#ifndef RAD_REF_DLOOP_FULL_H
#define RAD_REF_DLOOP_FULL_H

/* The host oracle publishes everything its row needs through dloop_full.h: the count, the rows and
 * the three launches. This header exists so a caller that links the reference plugin does not have
 * to know that, and so the include list of a test reads the way the plugin it tests does. */
#include "dloop_full.h"

#endif
