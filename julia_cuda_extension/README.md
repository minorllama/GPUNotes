# Writing CUDA C extensions for Julia

The `CUDA.jl` is required; if not available, `import Pkg; Pkg.add("CUDA")`. 

Now `make libcppexts.so` will build the extension, `/usr/bin/nvcc -O3 -Xcompiler -fPIC -shared libcppexts.cu -o libcppexts.so`. The path to `nvcc` may need to be updated in the `Makefile`. 

To kick the tires, start with `julia ./test_libcppexts.jl`. 

This was initially written before there was no support for `cuBLASDx` in Julia. [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) is a better option without descending into `CUDA C`.  

There are some delicate points here. The cpp driver function just calls `__global__` function - but there is no CUcontext that was setup in cpp, everything is under what was set up by julia, when it allocated CuArray to pass to cppext. One needs to make sure that the code is running in the correct context. Running uder julia, it should set up a thread local cuda context context (including device) which is what the code runs under (it's a call into so library: it runs in same process, which is set up with cuda by julia). 

The CUDA `libcppexts.cu` does not poll for errors. CUDA must explicitly be polled for errors. [NVIDIA's programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model) is very useful.  






