// Standalone GPU regression: hipcc --offload-arch=gfx1201 -I. tests/drafter_kv_compact_gpu.hip -o /tmp/drafter-kv-gpu
#include "arch/common/rad_kv_compact.h"
#include <hip/hip_runtime.h>
#include <cstdio>
int main(){
 for(int dev=0;dev<2;++dev){
  if(hipSetDevice(dev)!=hipSuccess)return 1;
  unsigned char original[256],actual[256];for(int i=0;i<256;++i)original[i]=(i*17+dev)%251;
  void* device=nullptr;hipStream_t stream;
  if(hipMalloc(&device,256)!=hipSuccess||hipStreamCreate(&stream)!=hipSuccess)return 2;
  if(hipMemcpy(device,original,256,hipMemcpyHostToDevice)!=hipSuccess)return 3;
  radiance::RadKVSlotMove moves[2]{};moves[0].src_slot=3;moves[0].dst_slot=1;moves[1].src_slot=5;moves[1].dst_slot=2;
  int rc=radiance::rad_kv_compact_winning_branch(device,128,64,32,4,8,2,2,moves,2,stream);
  if(rc||hipStreamSynchronize(stream)!=hipSuccess)return 4;
  if(hipMemcpy(actual,device,256,hipMemcpyDeviceToHost)!=hipSuccess)return 5;
  for(int l=0;l<2;++l)for(int h=0;h<2;++h)for(auto m:moves)for(int b=0;b<8;++b){
   int dst=l*128+(m.dst_slot/4)*64+h*32+(m.dst_slot%4)*8+b;
   int src=l*128+(m.src_slot/4)*64+h*32+(m.src_slot%4)*8+b;
   if(actual[dst]!=original[src]){std::printf("FAIL GPU%d byte%d\n",dev,dst);return 6;}
  }
  if(hipFree(device)!=hipSuccess || hipStreamDestroy(stream)!=hipSuccess)return 7;std::printf("PASS GPU%d compaction across block boundary, 2 layers, 2 heads\n",dev);
 }
}
