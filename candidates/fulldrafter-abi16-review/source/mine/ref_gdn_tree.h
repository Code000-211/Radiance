#ifndef RAD_REF_GDN_TREE_H
#define RAD_REF_GDN_TREE_H
#include "rad_abi.h"
#include "tree_gdn.h"

int rad_ref_gdn_conv_tree(const RadArgs*, RadStream);
int rad_ref_gdn_recurrent_tree(const RadArgs*, RadStream);
int rad_ref_gdn_tree_commit(const RadArgs*, RadStream);
int64_t rad_ref_gdn_tree_fused_scratch(const RadArgs*);
int rad_ref_gdn_tree_count(void);
const RadKernelInfo* rad_ref_gdn_tree_at(int);
#endif
