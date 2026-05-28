# GPUImprov

Small examples of writing GPU kernels in Python, Julia and C 

* Writing Python extensions using [Rust/burn](https://github.com/tracel-ai/burn): [Burn RK4 Example for SHO](burn_rk4/README.md).
* Writing CUDA extensions for Julia using CUDA.jl to set up context for a CUDA C extension: [Julia, CUDA.jl, CUDA C](julia_cuda_extension/README.md). 
This was initially written before there was no support for `cuBLASDx` in Julia. [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) is a better option without descending into `CUDA C`.
* Getting started with NVIDIA's experimental Rust to CUDA compiler, [cuda-oxide](https://github.com/NVlabs/cuda-oxide)  : [Setup](setup_cuda_oxide/README.md). 

