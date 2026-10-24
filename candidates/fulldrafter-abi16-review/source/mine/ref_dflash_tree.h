#ifndef RAD_REF_DFLASH_TREE_H
#define RAD_REF_DFLASH_TREE_H
#include "rad_abi.h"

/* The independent host implementation of `dflash_tree_select`. Same op definition, same validated
 * call struct, and its own tree construction -- see mine/ref_dflash_tree.cpp for what it shares
 * with the device row (arithmetic and order) and what it does not (bookkeeping). */
int rad_ref_dflash_tree_select(const RadArgs*, RadStream);
int rad_ref_dflash_tree_select_count(void);
const RadKernelInfo* rad_ref_dflash_tree_select_at(int);
#endif
