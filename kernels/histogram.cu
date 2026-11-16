#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

// Nonnegative finite float32 values are monotonically ordered by their stored bits.
constexpr int bits_per_page = 20;
struct Counts { unsigned fg, bg; double region; };
__global__ void count_pixels(const float* score, const unsigned char* truth,
    const double* weight, int pixels, Counts* bins, int first_page, int pages,
    double* by_image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    unsigned key = __float_as_uint(score[i]);
    int page = key >> bits_per_page;
    if (page < first_page || page >= first_page + pages) return;
    auto& cell = bins[key - ((unsigned)first_page << bits_per_page)];
    bool positive = truth[i];
    if (positive) {
        atomicAdd(&cell.fg, 1u);
        atomicAdd(&cell.region, weight[i]);
    } else atomicAdd(&cell.bg, 1u);
    if (by_image) {
        int bin = min(int(score[i] * 4096), 4095);
        auto dest = by_image + (i / (518*518)) * (3*4096) + bin;
        atomicAdd(dest + (positive ? 0 : 4096), 1.);
        if (positive) atomicAdd(dest + 8192, weight[i]);
    }
}
extern "C" void add_pixels(const float* score, const unsigned char* truth,
    const double* weight, int pixels, Counts* bins, int first_page, int pages,
    double* by_image, cudaStream_t stream) {
    count_pixels<<<(pixels+255)/256,256,0,stream>>>(score,truth,weight,pixels,bins,first_page,pages,by_image);
}
