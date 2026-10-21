#ifndef RAD_REF_PLE_TREE_H
#define RAD_REF_PLE_TREE_H
#include "ple_tree.h"

int rad_ref_ngram_ids_tree(const RadArgs*, RadStream);
int rad_ref_ple_conv_tree(const RadArgs*, RadStream);
int rad_ref_ple_tree_commit(const RadArgs*, RadStream);
int rad_ref_ple_tree_count(void);
const RadKernelInfo* rad_ref_ple_tree_at(int);
#endif
