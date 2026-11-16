NVCC ?= nvcc
CUDA_ARCH ?= sm_86
KERNEL_FLAGS := -arch=$(CUDA_ARCH) -O3 -shared -Xcompiler -fPIC

.PHONY: kernels
kernels: build/blur.so build/histogram.so

build/%.so: kernels/%.cu
	mkdir -p build
	$(NVCC) $(KERNEL_FLAGS) $< -o $@
