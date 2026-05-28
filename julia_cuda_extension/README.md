# Writing CUDA C extensions for Julia

`CUDA.jl` is required; if not available, `import Pkg; Pkg.add("CUDA")`. 

Now `make libcppexts.so` will build the extension, `/usr/bin/nvcc -O3 -Xcompiler -fPIC -shared libcppexts.cu -o libcppexts.so`. The path to `nvcc` may need to be updated in the `Makefile`. The CUDA code calls into a `device` function which uses device specific instructions as an example.

To kick the tires, start with `julia ./test_libcppexts.jl`. 

This was initially written before there was support for `cuBLASDx` in Julia, so to write fused kernels one had to descend into `CUDA C`; now [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) is a better option.   

There are some delicate points here: CuContext was set up by Julia, not `C++`. The cpp driver function just calls `__global__` function - but everything is with reference to the Julia context created when it allocated CuArray to pass to cppext. One needs to make sure that the context is correct Running uder julia, it should set up a thread local CUDA context (including device) which is what the code runs under (it's a call into `so` library: it runs in same process, which is set up with cuda by Julia). 

The CUDA `libcppexts.cu` does not poll for errors. CUDA must explicitly be polled for errors. [NVIDIA's programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model) is very useful.

## Loading ptx and cubin

It's useful to see how to load `ptx` file; it's possible to do some linking shenanigans and get some things to work that otherwise may not, like trying to dynamically link and call into a `device` function from a julia CUDA kernel. And there's no `so` shim. The CUDA code for this example is the usual `vadd.cu`. Trying out, `nvcc -ptx vadd.cu -o vadd.ptx`, 
```julia
julia> using CUDA
julia> m = CuModule(read("vadd.ptx"))
CuModule(Ptr{CUDACore.CUmod_st}(0x0000000015f36c30), CuContext(0x0000000015408160))
julia> f = CuFunction(m, "vadd")
CuFunction(Ptr{CUDACore.CUfunc_st}(0x0000000015f35ce0), CuModule(Ptr{CUDACore.CUmod_st}(0x0000000015f36c30), CuContext(0x0000000015408160)))
julia> n::Int32 = 4;  a = CUDA.ones(Float32, n); b = 2*CUDA.ones(Float32, n); c  = CUDA.zeros(Float32, n);
julia> CUDA.cudacall(f, (CuPtr{Float32}, CuPtr{Float32}, CuPtr{Float32}, Int32), a, b, c, n; threads=4); CUDA.synchronize(); c
4-element CuArray{Float32, 1, CUDACore.DeviceMemory}:
 3.0
 3.0
 3.0
 3.0

```
`cubin` works the same, other than needing to specify the architecture: `nvcc -cubin -arch=$arch vadd.cu -o vadd.cubin`. 
```julia
julia> using CUDA
julia> m = CuModuleFile("vadd.cubin")
CuModule(Ptr{CUDACore.CUmod_st}(0x00000000104456f0), CuContext(0x000000000f893b80))
julia> n::Int32 = 4;  a = CUDA.ones(Float32, n); b = 2*CUDA.ones(Float32, n); c  = CUDA.zeros(Float32, n);
julia> f = CuFunction(m, "vadd")
CuFunction(Ptr{CUDACore.CUfunc_st}(0x0000000010446040), CuModule(Ptr{CUDACore.CUmod_st}(0x00000000104456f0), CuContext(0x000000000f893b80)))
julia> CUDA.cudacall(f, (CuPtr{Float32}, CuPtr{Float32}, CuPtr{Float32}, Int32), a, b, c, n; threads=4); CUDA.synchronize(); c
4-element CuArray{Float32, 1, CUDACore.DeviceMemory}:
 3.0
 3.0
 3.0
 3.0
```







