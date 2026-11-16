#include <cuda_runtime.h>
__device__ int mirror(int x){return x<0 ? -1-x : x>=518 ? 1035-x : x;}
__global__ void smooth(const float *p,float *q,int n,int axis,const double*w){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n) return;
 int j=axis ? i%518 : (i/518)%518;
 int st=axis ? 1 : 518;
 double v=(double)p[i]*w[0];
 for(int k=16;k>0;k--) v+=((double)p[i+(mirror(j-k)-j)*st]+(double)p[i+(mirror(j+k)-j)*st])*w[k];
 q[i]=(float)v;
}
extern "C" void filter(const float *p,float *tmp,float *q,int n,const double*w,cudaStream_t stream){
 smooth<<<(n+255)/256,256,0,stream>>>(p,tmp,n,0,w);
 smooth<<<(n+255)/256,256,0,stream>>>(tmp,q,n,1,w);
}
